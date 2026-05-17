from dataclasses import dataclass, asdict
import argparse
import os

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers import WhisperTokenizer
import wandb
from jiwer import wer, cer
from tqdm import tqdm
import boto3


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):
        return x + self.pe[:, :x.size(1), :]

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class AudioEncoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, num_layers=8, dropout=0.1, n_mels=80):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

        self.pos_encoding = SinusoidalPE(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x):
        
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))

        x = x.permute(0,2,1)

        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, enc_out, causal_mask=None):
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x, _ = self.cross_attn(x, enc_out, enc_out)
        x = residual + x

        residual = x
        x = self.layer_norm3(x)
        x = self.ffn(x)
        x = residual + x
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, num_layers=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPE(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out):
        x = self.token_emb(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        for layer in self.layers:
            x = layer(x, enc_out, causal_mask)
        
        x = self.ln_final(x)
        logits = self.output_proj(x)
        return logits

class MiniWhisper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AudioEncoder(config.encoder_d_model, config.encoder_n_heads, config.encoder_d_ff, config.encoder_num_layers, config.dropout, config.n_mels)
        self.decoder = Decoder(config.vocab_size, config.decoder_d_model, config.decoder_n_heads, config.decoder_d_ff, config.decoder_num_layers, config.dropout)

    def forward(self, mel, tokens):
        enc_out = self.encoder(mel)
        logits = self.decoder(tokens, enc_out)
        return logits

@dataclass
class MiniWhisperConfig:
    encoder_d_model: int = 512
    encoder_n_heads: int = 8
    encoder_d_ff: int = 2048
    encoder_num_layers: int = 8

    decoder_d_model: int = 512
    decoder_n_heads: int = 8
    decoder_d_ff: int = 2048
    decoder_num_layers: int = 6

    n_mels: int = 80
    vocab_size: int = 51865
    dropout: float = 0.1

    
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_steps: int = 50000
    warmup_steps: int = 1000
    eval_every: int = 1000
    eval_batches: int = 500
    grad_clip: float = 1.0

class WhisperStreamDataset(IterableDataset):
    def __init__(self, stream, tokenizer, sample_rate=16000, n_mels=80, max_audio_len=480000, max_tokens=256):
        self.stream = stream
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len
        self.max_tokens = max_tokens

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        self.amp_to_db = AmplitudeToDB()

        self.SOT = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.EOT = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.HI = tokenizer.convert_tokens_to_ids("<|hi|>")
        self.TRANSLATE = tokenizer.convert_tokens_to_ids("<|translate|>")

    def __iter__(self):
        for sample in self.stream:
            result = self.process(sample)
            if result is not None:
                yield result

    def process(self, sample):
        # The dataset has audio in 'audio_filepath' dict with 'array' key
        audio = sample["audio_filepath"]["array"]


        if len(audio) > self.max_audio_len or len(audio) < 1600:
            return None

        waveform = torch.tensor(audio, dtype=torch.float32)
        mel = self.mel_transform(waveform)
        mel = self.amp_to_db(mel)


        # Use 'transcription' field which has the Hindi text
        text = sample["transcription"]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) > self.max_tokens - 4:
            return None

        
        
        decoder_input = [self.SOT, self.HI, self.TRANSLATE] + token_ids
        labels = [self.HI, self.TRANSLATE] + token_ids + [self.EOT]

        return mel, decoder_input, labels


def collate_fn(batch):
    mels, input_ids, labels = zip(*batch)

    
    max_t = max(m.size(1) for m in mels)
    mel_padded = torch.stack([F.pad(m, (0, max_t - m.size(1))) for m in mels])

    
    max_s = max(len(ids) for ids in input_ids)
    input_padded = torch.zeros(len(batch), max_s, dtype=torch.long)
    label_padded = torch.full((len(batch), max_s), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        input_padded[i, :len(inp)] = torch.tensor(inp)
        label_padded[i, :len(lab)] = torch.tensor(lab)

    return mel_padded, input_padded, label_padded


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-Whisper Training")

    
    parser.add_argument("--encoder_d_model", type=int, default=512)
    parser.add_argument("--encoder_n_heads", type=int, default=8)
    parser.add_argument("--encoder_d_ff", type=int, default=2048)
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_d_model", type=int, default=512)
    parser.add_argument("--decoder_n_heads", type=int, default=8)
    parser.add_argument("--decoder_d_ff", type=int, default=2048)
    parser.add_argument("--decoder_num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    
    parser.add_argument("--name", type=str, default="mini-whisper")
    parser.add_argument("--project", type=str, default="mini-whisper")

    return parser.parse_args()


def evaluate(model, val_loader, tokenizer, config, max_batches=500, device="cuda"):
    model.eval()
    total_loss = 0
    total_wer = 0
    total_cer = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating", total=max_batches)):
            if i >= max_batches:
                break

            mels, input_ids, labels = batch
            mels = mels.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(mels, input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )

            total_loss += loss.item()

            
            pred_ids = logits[0].argmax(dim=-1).cpu().numpy()
            label_ids = labels[0].cpu().numpy()

            
            label_ids = label_ids[label_ids != -100]

            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            truth_text = tokenizer.decode(label_ids, skip_special_tokens=True)

            if truth_text.strip():  
                total_wer += wer(truth_text, pred_text)
                total_cer += cer(truth_text, pred_text)

    n = min(max_batches, i + 1)
    return {
        "loss": total_loss / n,
        "wer": total_wer / n,
        "cer": total_cer / n
    }


if __name__ == "__main__":
    args = parse_args()

    config = MiniWhisperConfig(
        encoder_d_model=args.encoder_d_model,
        encoder_n_heads=args.encoder_n_heads,
        encoder_d_ff=args.encoder_d_ff,
        encoder_num_layers=args.encoder_num_layers,
        decoder_d_model=args.decoder_d_model,
        decoder_n_heads=args.decoder_n_heads,
        decoder_d_ff=args.decoder_d_ff,
        decoder_num_layers=args.decoder_num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        grad_clip=args.grad_clip
    )

    print(f"Config: {asdict(config)}")

    # Initialize W&B (optional - skip if no API key)
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    use_wandb = len(wandb_api_key) > 0

    if use_wandb:
        wandb.init(
            project=args.project,
            name=args.name,
            config=asdict(config)
        )
        print("✓ W&B logging enabled")
    else:
        wandb.init(mode="disabled")
        print("⚠ W&B disabled (no API key found)")


    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        print("⚠ Warning: HF_TOKEN not set. You may hit rate limits or fail to access gated datasets.")

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", token=hf_token)
    # Use len(tokenizer) instead of vocab_size to include special tokens (language, task tokens)
    config.vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size (base): {tokenizer.vocab_size}")
    print(f"Tokenizer vocab size (with special tokens): {len(tokenizer)}")
    print(f"Config vocab size: {config.vocab_size}")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniWhisper(config).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


    print("Loading dataset splits from S3...")
    s3_path = "s3://home-ml/stt_datasets"

    train_stream = load_dataset("arrow", data_files=f"{s3_path}/train/*.arrow", split="train", streaming=True).shuffle(buffer_size=5000)
    val_stream = load_dataset("arrow", data_files=f"{s3_path}/validation/*.arrow", split="train", streaming=True)
    test_stream = load_dataset("arrow", data_files=f"{s3_path}/test/*.arrow", split="train", streaming=True)

    train_ds = WhisperStreamDataset(train_stream, tokenizer)
    val_ds = WhisperStreamDataset(val_stream, tokenizer)
    test_ds = WhisperStreamDataset(test_stream, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, collate_fn=collate_fn)

    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps - config.warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps]
    )

    scaler = torch.amp.GradScaler("cuda")

    output_dir = os.environ.get("SM_MODEL_DIR", "./checkpoints")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # S3 setup for real-time checkpoint uploads
    s3_client = boto3.client('s3')
    s3_bucket = "home-ml"
    s3_prefix = f"experiments/{args.name}/checkpoints"
    print(f"Checkpoints will be synced to: s3://{s3_bucket}/{s3_prefix}/")

    print("Starting training...")
    model.train()
    global_step = 0
    running_loss = 0

    pbar = tqdm(total=config.max_steps, desc="Training")

    while global_step < config.max_steps:
        for batch in train_loader:
            mels, input_ids, labels = batch
            mels = mels.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(mels, input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            pbar.update(1)

            if global_step % 100 == 0:
                avg_loss = running_loss / 100
                wandb.log({
                    "train_loss": avg_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step
                })
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0

            if global_step % config.eval_every == 0:
                print(f"\nEvaluating at step {global_step}...")
                metrics = evaluate(model, val_loader, tokenizer, config, max_batches=config.eval_batches, device=device)

                print(f"Val Loss: {metrics['loss']:.4f}, WER: {metrics['wer']*100:.2f}%, CER: {metrics['cer']*100:.2f}%")

                wandb.log({
                    "val_loss": metrics["loss"],
                    "val_wer": metrics["wer"],
                    "val_cer": metrics["cer"],
                    "step": global_step
                })

                checkpoint_path = f"{checkpoint_dir}/{args.name}_step_{global_step}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": asdict(config),
                    "step": global_step,
                    "val_metrics": metrics
                }, checkpoint_path)
                print(f"Checkpoint saved locally: {checkpoint_path}")

                # Upload to S3 immediately
                try:
                    s3_key = f"{s3_prefix}/{args.name}_step_{global_step}.pt"
                    s3_client.upload_file(checkpoint_path, s3_bucket, s3_key)
                    print(f"✓ Uploaded to s3://{s3_bucket}/{s3_key}")
                except Exception as e:
                    print(f"⚠ S3 upload failed: {e}")

                model.train()

            if global_step >= config.max_steps:
                break

    pbar.close()

    print("\n" + "="*50)
    print("Running final test evaluation...")
    test_metrics = evaluate(model, test_loader, tokenizer, config, max_batches=config.eval_batches, device=device)

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test WER: {test_metrics['wer']*100:.2f}%")
    print(f"Test CER: {test_metrics['cer']*100:.2f}%")

    wandb.log({
        "test_loss": test_metrics["loss"],
        "test_wer": test_metrics["wer"],
        "test_cer": test_metrics["cer"]
    })

    final_path = f"{checkpoint_dir}/{args.name}_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "step": global_step,
        "test_metrics": test_metrics
    }, final_path)
    print(f"Final model saved locally: {final_path}")

    # Upload final checkpoint to S3
    try:
        s3_key = f"{s3_prefix}/{args.name}_final.pt"
        s3_client.upload_file(final_path, s3_bucket, s3_key)
        print(f"✓ Final model uploaded to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"⚠ S3 upload failed: {e}")

    wandb.finish()

