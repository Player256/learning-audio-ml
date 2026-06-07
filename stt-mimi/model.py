"""
Qwen3 with Cross-Attention for Audio-to-Text STT
Simple, clean implementation for training

Audio (Mimi codes) → Audio Encoder → Cross-Attention → Qwen3 Decoder → Text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# Basic Building Blocks
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class MLP(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys."""
    # cos, sin: [B, 1, T, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # k might have different seq len than q
    k_cos = cos[:, :, :k.shape[2], :]
    k_sin = sin[:, :, :k.shape[2], :]
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings."""
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [B, num_heads, seq_len, head_dim]
        # position_ids: [B, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


# ============================================================================
# Attention Modules
# ============================================================================

def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads for Grouped Query Attention."""
    if n_rep == 1:
        return hidden_states
    B, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(B, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """Multi-head attention with optional cross-attention and RoPE."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, is_cross_attn=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim  # Use explicit head_dim (may be larger than hidden_size // num_heads)
        self.num_kv_groups = num_heads // num_kv_heads
        self.is_cross_attn = is_cross_attn
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=True)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_embeddings=None):
        B, T, _ = hidden_states.shape

        # Queries from decoder
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, num_heads, T, head_dim]

        # Keys/Values from encoder (cross-attn) or same sequence (self-attn)
        kv_input = key_value_states if self.is_cross_attn else hidden_states
        T_kv = kv_input.shape[1]

        k = self.k_proj(kv_input).view(B, T_kv, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_input).view(B, T_kv, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE (only for self-attention)
        if position_embeddings is not None and not self.is_cross_attn:
            cos, sin = position_embeddings
            cos = cos.unsqueeze(1)  # [B, 1, T, head_dim]
            sin = sin.unsqueeze(1)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand k/v for GQA
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn_output)


# ============================================================================
# Transformer Layer
# ============================================================================

class DecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention, cross-attention, and MLP."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, head_dim):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, is_cross_attn=False)
        self.cross_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, is_cross_attn=True)
        self.mlp = MLP(hidden_size, intermediate_size)

        self.input_norm = RMSNorm(hidden_size)
        self.cross_attn_norm = RMSNorm(hidden_size)
        self.post_attn_norm = RMSNorm(hidden_size)

    def forward(self, hidden_states, encoder_hidden_states=None,
                attention_mask=None, encoder_attention_mask=None, position_embeddings=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask,
                                       position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        # Cross-attention (if encoder states provided)
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states = self.cross_attn(hidden_states, key_value_states=encoder_hidden_states,
                                           attention_mask=encoder_attention_mask)
            hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class AudioEncoder(nn.Module):
    def __init__(self, num_codebooks, codebook_size, hidden_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size) for _ in range(num_codebooks)
        ])
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, audio_codes):
        """
        Args:
            audio_codes: [B, num_codebooks, T]
        Returns:
            audio_context: [B, T, hidden_size]
        """
        B, K, T = audio_codes.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"
        embeddings = torch.stack([self.embeddings[k](audio_codes[:, k]) for k in range(K)])
        audio_context = embeddings.sum(dim=0)  
        return self.proj(audio_context)


class Qwen3STT(nn.Module):
    """Qwen3 with cross-attention(text tokens as querying over audio tokens)"""

    def __init__(self, vocab_size=151936, hidden_size=1024, num_layers=28,
                 num_heads=16, num_kv_heads=8, intermediate_size=3072,
                 max_position_embeddings=32768, num_audio_codebooks=32,
                 audio_codebook_size=2048, head_dim=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim if head_dim is not None else (hidden_size // num_heads)

        self.audio_encoder = AudioEncoder(num_audio_codebooks, audio_codebook_size, hidden_size)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size, self.head_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)

        print(f"Model initialized: {sum(p.numel() for p in self.parameters()):,} total parameters")

    def forward(self, input_ids, audio_codes, labels=None,
                attention_mask=None, audio_attention_mask=None):
        """
        Args:
            input_ids: [B, S] - text token IDs
            audio_codes: [B, K, T] - Mimi audio codes
            labels: [B, S] - text labels for loss (optional)
            attention_mask: [B, S] - text padding mask (optional)
            audio_attention_mask: [B, T] - audio padding mask (optional)
        """
        B, S = input_ids.shape
        device = input_ids.device

        encoder_hidden_states = self.audio_encoder(audio_codes) 

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = self._make_causal_mask(S, hidden_states.dtype, device)
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
            padding_mask = (1.0 - padding_mask) * torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + padding_mask

        encoder_mask = None
        if audio_attention_mask is not None:
            encoder_mask = audio_attention_mask[:, None, None, :].to(hidden_states.dtype)
            encoder_mask = (1.0 - encoder_mask) * torch.finfo(hidden_states.dtype).min

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=causal_mask,
                encoder_attention_mask=encoder_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}

    @staticmethod
    def _make_causal_mask(seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen3-0.6B", **kwargs):
        """Load pretrained Qwen weights and initialize model."""
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading pretrained weights from {model_name}...")
        config = AutoConfig.from_pretrained(model_name)

        # Get head_dim from config (Qwen3 uses explicit head_dim that may be larger than hidden_size // num_heads)
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

        print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, "
              f"num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}, "
              f"intermediate_size={config.intermediate_size}, head_dim={head_dim}")

        pretrained = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        # Create model with matching config
        model = cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=head_dim,  # Pass explicit head_dim
            **kwargs
        )

        # Copy pretrained weights
        print(f"Copying weights (model hidden_size={model.hidden_size}, "
              f"pretrained hidden_size={pretrained.config.hidden_size})...")
        model.embed_tokens.weight.data.copy_(pretrained.model.embed_tokens.weight.data)
        model.norm.weight.data.copy_(pretrained.model.norm.weight.data)
        model.lm_head.weight.data.copy_(pretrained.lm_head.weight.data)

        for i, (our_layer, pre_layer) in enumerate(zip(model.layers, pretrained.model.layers)):
            # Self-attention
            our_layer.self_attn.q_proj.weight.data.copy_(pre_layer.self_attn.q_proj.weight.data)
            our_layer.self_attn.k_proj.weight.data.copy_(pre_layer.self_attn.k_proj.weight.data)
            our_layer.self_attn.v_proj.weight.data.copy_(pre_layer.self_attn.v_proj.weight.data)
            our_layer.self_attn.o_proj.weight.data.copy_(pre_layer.self_attn.o_proj.weight.data)

            # Copy biases if they exist
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                our_proj = getattr(our_layer.self_attn, proj_name)
                pre_proj = getattr(pre_layer.self_attn, proj_name)
                if hasattr(pre_proj, 'bias') and pre_proj.bias is not None:
                    our_proj.bias.data.copy_(pre_proj.bias.data)

            # Q/K norms (if they exist in pretrained)
            if hasattr(pre_layer.self_attn, 'q_norm'):
                our_layer.self_attn.q_norm.weight.data.copy_(pre_layer.self_attn.q_norm.weight.data)
            if hasattr(pre_layer.self_attn, 'k_norm'):
                our_layer.self_attn.k_norm.weight.data.copy_(pre_layer.self_attn.k_norm.weight.data)

            # MLP
            our_layer.mlp.gate_proj.weight.data.copy_(pre_layer.mlp.gate_proj.weight.data)
            our_layer.mlp.up_proj.weight.data.copy_(pre_layer.mlp.up_proj.weight.data)
            our_layer.mlp.down_proj.weight.data.copy_(pre_layer.mlp.down_proj.weight.data)

            # Layer norms
            our_layer.input_norm.weight.data.copy_(pre_layer.input_layernorm.weight.data)
            our_layer.post_attn_norm.weight.data.copy_(pre_layer.post_attention_layernorm.weight.data)

        print("✅ Weights loaded successfully")
        del pretrained  # Free memory
        return model

    def freeze_pretrained(self):
        """Freeze all pretrained weights, keep only cross-attention trainable."""
        # Freeze embeddings and output
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

        # Freeze pretrained parts of each layer
        for layer in self.layers:
            for param in layer.self_attn.parameters():
                param.requires_grad = False
            for param in layer.mlp.parameters():
                param.requires_grad = False
            for param in layer.input_norm.parameters():
                param.requires_grad = False
            for param in layer.post_attn_norm.parameters():
                param.requires_grad = False
            # cross_attn and cross_attn_norm stay trainable

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"✅ Frozen: {frozen:,} params | Trainable: {trainable:,} params ({100*trainable/(trainable+frozen):.1f}%)")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Initializing model...\n")
    model = Qwen3STT.from_pretrained(
        "Qwen/Qwen3-0.6B",
        num_audio_codebooks=32,
        audio_codebook_size=2048,
    )

    print("\nFreezing pretrained weights...\n")
    model.freeze_pretrained()

    print("\nTesting forward pass...")
    B, S, T = 2, 50, 250
    input_ids = torch.randint(0, 151936, (B, S))
    audio_codes = torch.randint(0, 2048, (B, 32, T))
    labels = torch.randint(0, 151936, (B, S))

    outputs = model(input_ids=input_ids, audio_codes=audio_codes, labels=labels)

    print(f"✅ Loss: {outputs['loss'].item():.4f}")
    print(f"✅ Logits shape: {outputs['logits'].shape}")
    print("\n✨ Model ready for training!")
