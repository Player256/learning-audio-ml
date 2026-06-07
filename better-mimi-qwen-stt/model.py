import os
import io
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel, Trainer, TrainingArguments
from datasets import load_dataset
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from jiwer import wer, cer

mimi = MimiModel.from_pretrained("kyutai/mimi")

apply_liger_kernel_to_qwen3()
qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

for param in qwen.parameters():
    param.requires_grad = False

DecoderLayerClass = qwen.model.layers[0].__class__
RMSNormClass = qwen.model.layers[0].input_layernorm.__class__


class CrossAttentionModule(DecoderLayerClass):
    def __init__(self):
        pass

    def inject_cross_attn(self, hidden_size):
        self.cross_attn_norm = RMSNormClass(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=16, batch_first=True
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        encoder_output = getattr(self, "_encoder_output", None)
        if encoder_output is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states, _ = self.cross_attn(
                query=hidden_states,
                key=encoder_output,
                value=encoder_output,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


for layer in qwen.model.layers:
    layer.__class__ = CrossAttentionModule
    layer.inject_cross_attn(qwen.config.hidden_size)


MAX_AUDIO_SAMPLES = 480000  # 30s at 16kHz


class QwenMimiSTT(nn.Module):
    def __init__(self, qwen, tokenizer, mimi, num_quantizers_to_use=8):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.mimi = mimi
        self.num_quantizers_to_use = num_quantizers_to_use
        self._mimi_device_set = False

        self.code_embeddings = nn.ModuleList()
        semantic_layers = mimi.quantizer.semantic_residual_vector_quantizer.layers
        acoustic_layers = mimi.quantizer.acoustic_residual_vector_quantizer.layers
        all_layers = list(semantic_layers) + list(acoustic_layers)
        for q in range(num_quantizers_to_use):
            codebook = all_layers[q].codebook
            emb = nn.Embedding(mimi.config.codebook_size, mimi.config.codebook_dim)
            emb.weight.data.copy_(codebook.embed)
            emb.weight.requires_grad = False
            self.code_embeddings.append(emb)

        self.projector = nn.Linear(
            mimi.config.codebook_dim,
            qwen.config.hidden_size,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.qwen.gradient_checkpointing_disable()

    def _ensure_mimi_on_device(self, device):
        if not self._mimi_device_set:
            self.mimi.to(device)
            for layer in self.mimi.quantizer.semantic_residual_vector_quantizer.layers:
                layer.codebook._embed = None
            for layer in self.mimi.quantizer.acoustic_residual_vector_quantizer.layers:
                layer.codebook._embed = None
            self._mimi_device_set = True

    def encode_audio(self, audio_codes):
        combined = torch.zeros(
            audio_codes.shape[0], audio_codes.shape[2], self.mimi.config.codebook_dim,
            device=audio_codes.device, dtype=self.projector.weight.dtype,
        )
        for q in range(self.num_quantizers_to_use):
            combined = combined + self.code_embeddings[q](audio_codes[:, q, :])

        return self.projector(combined)

    def forward(self, input_ids, waveforms, attention_mask=None, labels=None, **kwargs):
        with torch.no_grad():
            self._ensure_mimi_on_device(input_ids.device)
            audio_codes = self.mimi.encode(waveforms.unsqueeze(1)).audio_codes

        encoder_output = self.encode_audio(audio_codes)

        for layer in self.qwen.model.layers:
            layer._encoder_output = encoder_output

        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}


model = QwenMimiSTT(qwen, tokenizer, mimi)

train_dataset = load_dataset("parquet", data_files="s3://home-ml/asr-dataset/train/*", streaming=True)["train"]
val_dataset = load_dataset("parquet", data_files="s3://home-ml/asr-dataset/val/*", streaming=True)["train"]


def collate_fn(batch):
    waveforms = []
    for item in batch:
        audio_array, _ = sf.read(io.BytesIO(item["audio_bytes"]))
        waveform = torch.tensor(audio_array, dtype=torch.float32)
        if waveform.shape[0] > MAX_AUDIO_SAMPLES:
            waveform = waveform[:MAX_AUDIO_SAMPLES]
        waveforms.append(waveform)

    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.stack([
        nn.functional.pad(w, (0, max_len - w.shape[0]))
        for w in waveforms
    ])

    transcripts = [item["text"] for item in batch]
    tokens = tokenizer(
        transcripts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "waveforms": padded,
        "labels": labels,
    }


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    pad_id = tokenizer.pad_token_id or 0
    preds = np.where((preds >= 0) & (preds < tokenizer.vocab_size), preds, pad_id)
    labels = np.where((labels >= 0) & (labels < tokenizer.vocab_size), labels, pad_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() if p.strip() else " " for p in decoded_preds]
    decoded_labels = [l.strip() if l.strip() else " " for l in decoded_labels]

    return {
        "wer": wer(decoded_labels, decoded_preds),
        "cer": cer(decoded_labels, decoded_preds),
    }


class STTTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            waveforms=inputs["waveforms"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                waveforms=inputs["waveforms"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        loss = outputs["loss"]
        if prediction_loss_only:
            return (loss, None, None)
        preds = outputs["logits"][:, :-1, :].argmax(dim=-1)
        labels = inputs["labels"][:, 1:]
        return (loss, preds, labels)


run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
os.environ["WANDB_PROJECT"] = "qwen-mimi-better"

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=43500,
    learning_rate=1e-4,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    torch_compile=True,
    dataloader_num_workers=4,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    remove_unused_columns=False,
    report_to="wandb",
    run_name=run_name,
    eval_accumulation_steps=10,
    max_grad_norm=1.0,
)

trainer = STTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
