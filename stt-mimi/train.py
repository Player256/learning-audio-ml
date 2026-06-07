"""
Training script for Mimi + Qwen STT Model
Runs on SageMaker with mixed precision and gradient accumulation
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
from jiwer import wer, cer

from training_qwen_mimi import Qwen3STT
from custom_dataset import create_dataloaders


def get_args():
    """Parse hyperparameters from SageMaker or defaults."""
    args = {
        # Model
        'model_name': os.environ.get('SM_HP_MODEL_NAME', 'Qwen/Qwen3-0.6B'),
        'hidden_size': int(os.environ.get('SM_HP_HIDDEN_SIZE', '896')),
        'num_layers': int(os.environ.get('SM_HP_NUM_LAYERS', '24')),
        'num_heads': int(os.environ.get('SM_HP_NUM_HEADS', '14')),
        'num_kv_heads': int(os.environ.get('SM_HP_NUM_KV_HEADS', '2')),
        'intermediate_size': int(os.environ.get('SM_HP_INTERMEDIATE_SIZE', '4864')),

        # Training
        'batch_size': int(os.environ.get('SM_HP_BATCH_SIZE', '4')),
        'gradient_accumulation_steps': int(os.environ.get('SM_HP_GRAD_ACCUM', '4')),
        'lr': float(os.environ.get('SM_HP_LR', '1e-4')),
        'weight_decay': float(os.environ.get('SM_HP_WEIGHT_DECAY', '0.01')),
        'max_steps': int(os.environ.get('SM_HP_MAX_STEPS', '50000')),
        'warmup_steps': int(os.environ.get('SM_HP_WARMUP_STEPS', '1000')),
        'grad_clip': float(os.environ.get('SM_HP_GRAD_CLIP', '1.0')),

        # Data
        's3_data_path': os.environ.get('SM_HP_S3_DATA_PATH', 's3://home-ml/stt_datasets'),
        'num_workers': int(os.environ.get('SM_HP_NUM_WORKERS', '0')),

        # Logging
        'log_interval': int(os.environ.get('SM_HP_LOG_INTERVAL', '10')),
        'eval_interval': int(os.environ.get('SM_HP_EVAL_INTERVAL', '500')),
        'save_interval': int(os.environ.get('SM_HP_SAVE_INTERVAL', '2000')),
        'wandb_project': os.environ.get('SM_HP_WANDB_PROJECT', 'mimi-qwen-stt'),
        'run_name': os.environ.get('SM_HP_RUN_NAME', 'qwen-stt-small'),

        # Paths
        'output_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
        'checkpoint_dir': os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints'),
    }

    return args


def compute_metrics(logits, labels):
    """Compute accuracy metrics."""
    predictions = logits.argmax(dim=-1)
    mask = labels != -100

    correct = ((predictions == labels) & mask).sum()
    total = mask.sum()

    accuracy = correct.float() / total if total > 0 else torch.tensor(0.0)
    return {'accuracy': accuracy.item()}


def train_step(model, batch, scaler, grad_accum_step):
    """Single training step with mixed precision."""

    audio_codes = batch['audio_codes'].cuda()
    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()
    audio_mask = batch['audio_attention_mask'].cuda()
    text_mask = batch['attention_mask'].cuda()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            audio_codes=audio_codes,
            labels=labels,
            attention_mask=text_mask,
            audio_attention_mask=audio_mask,
        )
        loss = outputs['loss'] / grad_accum_step

    scaler.scale(loss).backward()

    metrics = compute_metrics(outputs['logits'], labels)
    metrics['loss'] = loss.item() * grad_accum_step

    return metrics


@torch.no_grad()
def eval_step(model, val_loader, tokenizer, max_eval_steps=100):
    """Evaluation loop with CER/WER metrics."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_wer = 0.0
    total_cer = 0.0
    num_steps = 0
    num_text_samples = 0

    for batch in tqdm(val_loader, desc="Evaluating", total=max_eval_steps):
        if num_steps >= max_eval_steps:
            break

        audio_codes = batch['audio_codes'].cuda()
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        audio_mask = batch['audio_attention_mask'].cuda()
        text_mask = batch['attention_mask'].cuda()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                audio_codes=audio_codes,
                labels=labels,
                attention_mask=text_mask,
                audio_attention_mask=audio_mask,
            )

        metrics = compute_metrics(outputs['logits'], labels)
        total_loss += outputs['loss'].item()
        total_acc += metrics['accuracy']

        pred_ids = outputs['logits'][0].argmax(dim=-1).cpu().numpy()
        label_ids = labels[0].cpu().numpy()

        label_ids = label_ids[label_ids != -100]

        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        truth_text = tokenizer.decode(label_ids, skip_special_tokens=True)

        if truth_text.strip():
            total_wer += wer(truth_text, pred_text)
            total_cer += cer(truth_text, pred_text)
            num_text_samples += 1

        num_steps += 1

    model.train()

    return {
        'val_loss': total_loss / num_steps,
        'val_accuracy': total_acc / num_steps,
        'val_wer': total_wer / num_text_samples if num_text_samples > 0 else 0.0,
        'val_cer': total_cer / num_text_samples if num_text_samples > 0 else 0.0,
    }


def save_checkpoint(model, optimizer, scheduler, step, args):
    """Save checkpoint to local and S3."""
    checkpoint_path = os.path.join(args['checkpoint_dir'], f"checkpoint-{step}.pt")
    os.makedirs(args['checkpoint_dir'], exist_ok=True)

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
    }, checkpoint_path)

    print(f"✅ Saved checkpoint: {checkpoint_path}")

    final_path = os.path.join(args['output_dir'], f"checkpoint-{step}.pt")
    os.makedirs(args['output_dir'], exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'args': args,
    }, final_path)


def main():
    args = get_args()

    print("="*60)
    print("Mimi + Qwen STT Training")
    print("="*60)
    for k, v in args.items():
        print(f"  {k}: {v}")
    print("="*60)

    # Initialize W&B
    wandb.init(
        project=args['wandb_project'],
        name=args['run_name'],
        config=args,
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, tokenizer, mimi = create_dataloaders(
        s3_path=args['s3_data_path'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
    )

    # Update vocab size if tokenizer was extended
    args['vocab_size'] = len(tokenizer)

    # Create model
    print("\nInitializing model...")
    model = Qwen3STT.from_pretrained(
        args['model_name'],
        num_audio_codebooks=32,
        audio_codebook_size=2048,
    ).cuda()

    # Resize embeddings if needed
    if len(tokenizer) != model.vocab_size:
        print(f"Resizing embeddings: {model.vocab_size} → {len(tokenizer)}")
        model.embed_tokens = torch.nn.Embedding(len(tokenizer), model.hidden_size).cuda()
        model.lm_head = torch.nn.Linear(model.hidden_size, len(tokenizer), bias=False).cuda()
        model.vocab_size = len(tokenizer)

    model.freeze_pretrained()
    model.train()

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args['lr'],
        weight_decay=args['weight_decay'],
        betas=(0.9, 0.999),
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args['max_steps'] - args['warmup_steps'],
        eta_min=args['lr'] * 0.1,
    )

    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args['warmup_steps'],
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda")

    # Training loop
    print("\nStarting training...\n")

    step = 0
    running_loss = 0.0
    running_acc = 0.0

    train_iter = iter(train_loader)

    while step < args['max_steps']:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        metrics = train_step(
            model, batch, scaler,
            grad_accum_step=args['gradient_accumulation_steps']
        )

        running_loss += metrics['loss']
        running_acc += metrics['accuracy']

        # Gradient accumulation
        if (step + 1) % args['gradient_accumulation_steps'] == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_clip'])

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler step
            if step < args['warmup_steps']:
                warmup_scheduler.step()
            else:
                scheduler.step()

        step += 1

        # Logging
        if step % args['log_interval'] == 0:
            avg_loss = running_loss / args['log_interval']
            avg_acc = running_acc / args['log_interval']
            lr = optimizer.param_groups[0]['lr']

            print(f"Step {step}/{args['max_steps']} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.3f} | "
                  f"LR: {lr:.2e}")

            wandb.log({
                'train/loss': avg_loss,
                'train/accuracy': avg_acc,
                'train/lr': lr,
                'step': step,
            })

            running_loss = 0.0
            running_acc = 0.0

        # Evaluation
        if step % args['eval_interval'] == 0 and step > 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at step {step}...")
            eval_metrics = eval_step(model, val_loader, tokenizer, max_eval_steps=100)

            print(f"Val Loss: {eval_metrics['val_loss']:.4f} | "
                  f"Val Acc: {eval_metrics['val_accuracy']:.3f} | "
                  f"WER: {eval_metrics['val_wer']*100:.2f}% | "
                  f"CER: {eval_metrics['val_cer']*100:.2f}%")
            print(f"{'='*60}\n")

            wandb.log({**eval_metrics, 'step': step})

        # Checkpointing
        if step % args['save_interval'] == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, args)

    # Final save
    print("\nTraining complete! Saving final model...")
    save_checkpoint(model, optimizer, scheduler, step, args)

    wandb.finish()
    print("✅ Done!")


if __name__ == "__main__":
    main()
