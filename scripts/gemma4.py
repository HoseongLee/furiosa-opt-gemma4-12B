"""
Gemma 4 (google/gemma-4-12B-it text decoder) — self-contained reference implementation
(pure PyTorch).

Architecture: Gemma4ForCausalLM, trimmed to exactly what this checkpoint's config.json
uses (confirmed against the real config, not the generic `Gemma4TextConfig()` defaults
in transformers — see also this repo's src/axes.rs and scripts/generate_references.py,
which independently derive the sliding-attention shapes from safetensors headers):
  - RMSNorm: plain rescale (x_hat * weight), weight initialized to ones — NOT the
    "(1 + weight)" zero-centered convention used by Gemma 2/3.
  - "Sandwich" norms: input/post-attention/pre-feedforward/post-feedforward RMSNorm
    around every attention and MLP sub-block (4 norms per layer, all pre-residual-add).
  - GeGLU MLP (gelu_pytorch_tanh), SwiGLU-shaped: down(gelu(gate(x)) * up(x)).
  - Grouped-Query Attention with per-head RMSNorm on q/k (scaled) and v (unscaled).
    Attention scaling is fixed at 1.0 (folded into the norms), not 1/sqrt(head_dim).
  - 48 layers, alternating 5 sliding : 1 full (last layer is full_attention). The two
    layer types are genuinely different geometries, not just a windowed-vs-not toggle:
      * sliding_attention (40 layers): head_dim=256, 8 KV heads (GQA, 2 query heads
        each), RoPE theta=10_000, fully rotated.
      * full_attention (8 layers, i%6==5): head_dim=512, a SINGLE KV head (true MQA —
        there is no v_proj at all; V is renormalized K), RoPE theta=1_000_000, only the
        first 25% of angles rotated ("proportional" RoPE — the rest is padded with
        zero frequency, i.e. identity).
  - Final logits are softcapped: logits = tanh(logits / 30) * 30.
  - Tied input/output embeddings; token embeddings scaled by sqrt(hidden_size).

Not implemented (present in the general Gemma4 architecture, but confirmed unused by
this checkpoint's config: no per-layer-embedding table, no MoE router/experts, no
KV-shared trailing layers, no double-wide MLP): per-layer-embeddings (PLE),
Mixture-of-Experts blocks, KV-layer sharing.

Deps: torch only (transformers optional, for weight/config loading).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Gemma4Config:
    vocab_size: int = 262_144
    hidden_size: int = 3840
    intermediate_size: int = 15_360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    max_position_embeddings: int = 262_144
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    layer_types: list = None

    local_rope_theta: float = 10_000.0
    global_rope_theta: float = 1_000_000.0
    global_partial_rotary_factor: float = 0.25

    attention_k_eq_v: bool = True
    num_global_key_value_heads: int = 1

    final_logit_softcapping: float = 30.0

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % self.sliding_window_pattern else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
            self.layer_types[-1] = "full_attention"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(dtype)


class ScaledEmbedding(nn.Embedding):
    """nn.Embedding whose output is multiplied by a fixed (non-persistent, dtype-following) scale."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


def compute_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float) -> torch.Tensor:
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2).float() / head_dim))
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles)])
    return inv_freq


def precompute_rope(head_dim: int, max_pos: int, theta: float, partial_rotary_factor: float, device, dtype):
    inv_freq = compute_inv_freq(head_dim, theta, partial_rotary_factor).to(device)
    pos = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


def sliding_window_mask(seq_len: int, window: int, device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    dist = idx[:, None] - idx[None, :]
    return (dist >= 0) & (dist < window)


class Attention(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = cfg.sliding_window if self.is_sliding else None

        self.head_dim = cfg.head_dim if self.is_sliding else cfg.global_head_dim
        self.use_alt_attn = cfg.attention_k_eq_v and not self.is_sliding
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_global_key_value_heads if self.use_alt_attn else cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = (
            None if self.use_alt_attn
            else nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=cfg.attention_bias)
        )
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False)

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rope(q, cos, sin).transpose(1, 2)

        k_raw = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v_raw = self.v_proj(x).view(B, T, self.n_kv, self.head_dim) if self.v_proj is not None else k_raw

        k = apply_rope(self.k_norm(k_raw), cos, sin).transpose(1, 2)
        v = self.v_norm(v_raw).transpose(1, 2)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        if self.is_sliding:
            mask = sliding_window_mask(T, self.sliding_window, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Gemma4Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, cos, sin):
        residual = x
        x = self.self_attn(self.input_layernorm(x), cos, sin)
        x = residual + self.post_attention_layernorm(x)

        residual = x
        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = residual + self.post_feedforward_layernorm(h)

        return x * self.layer_scalar


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, cfg: Gemma4Config = Gemma4Config()):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = ScaledEmbedding(
            cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, embed_scale=cfg.hidden_size**0.5
        )
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        _, T = input_ids.shape

        x = self.embed_tokens(input_ids)

        cos_sin_by_type = {
            "sliding_attention": precompute_rope(
                cfg.head_dim, cfg.max_position_embeddings, cfg.local_rope_theta, 1.0, x.device, x.dtype
            ),
            "full_attention": precompute_rope(
                cfg.global_head_dim, cfg.max_position_embeddings, cfg.global_rope_theta,
                cfg.global_partial_rotary_factor, x.device, x.dtype,
            ),
        }
        cos_sin_by_type = {k: (cos[:T], sin[:T]) for k, (cos, sin) in cos_sin_by_type.items()}

        for i, layer in enumerate(self.layers):
            cos, sin = cos_sin_by_type[cfg.layer_types[i]]
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        if cfg.final_logit_softcapping is not None:
            logits = torch.tanh(logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping

        return logits


def config_from_pretrained(model_id: str = "google/gemma-4-12B-it") -> Gemma4Config:
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(model_id)
    tc = getattr(hf_cfg, "text_config", hf_cfg)
    rope = tc.rope_parameters

    unsupported = {
        "hidden_size_per_layer_input (PLE)": tc.hidden_size_per_layer_input,
        "enable_moe_block": tc.enable_moe_block,
        "num_kv_shared_layers": tc.num_kv_shared_layers,
        "use_double_wide_mlp": tc.use_double_wide_mlp,
    }
    active = {name: val for name, val in unsupported.items() if val}
    if active:
        raise NotImplementedError(
            f"{model_id}'s config uses features this trimmed reference doesn't implement: {active}. "
            "It was written for google/gemma-4-12B-it, which has none of these active."
        )

    return Gemma4Config(
        vocab_size=tc.vocab_size,
        hidden_size=tc.hidden_size,
        intermediate_size=tc.intermediate_size,
        num_hidden_layers=tc.num_hidden_layers,
        num_attention_heads=tc.num_attention_heads,
        num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,
        global_head_dim=tc.global_head_dim,
        max_position_embeddings=tc.max_position_embeddings,
        rms_norm_eps=tc.rms_norm_eps,
        pad_token_id=tc.pad_token_id or 0,
        tie_word_embeddings=tc.tie_word_embeddings,
        attention_bias=tc.attention_bias,
        sliding_window=tc.sliding_window,
        layer_types=list(tc.layer_types),
        local_rope_theta=rope["sliding_attention"]["rope_theta"],
        global_rope_theta=rope["full_attention"]["rope_theta"],
        global_partial_rotary_factor=rope["full_attention"].get("partial_rotary_factor", 1.0),
        attention_k_eq_v=tc.attention_k_eq_v,
        num_global_key_value_heads=tc.num_global_key_value_heads or tc.num_key_value_heads,
        final_logit_softcapping=tc.final_logit_softcapping,
    )


def load_hf_weights(model: Gemma4ForCausalLM, model_id: str = "google/gemma-4-12B-it"):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    sd = {k.replace("model.", "", 1): v for k, v in hf.state_dict().items()}
    model.load_state_dict(sd, strict=False)
    return model


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_id = "google/gemma-4-12B-it"
    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = config_from_pretrained(model_id)
    model = Gemma4ForCausalLM(cfg).to(torch.bfloat16).eval()
    load_hf_weights(model, model_id)

    ids = tok("The capital of France is", return_tensors="pt").input_ids
    for _ in range(20):
        logits = model(ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    print(tok.decode(ids[0]))
