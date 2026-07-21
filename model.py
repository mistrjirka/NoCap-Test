"""Decoder-only Transformer variants for the BottleCapAI NoCap benchmark.

The repository keeps the public dense baseline, the Dense512 vocabulary
factorization experiment, and an optional LiquidLite hybrid. LiquidLite is
inspired by LFM2's gated short convolution operator, but deliberately keeps
most layers as ordinary global causal attention for a safer 124M-scale test.

The shared token matrix must inherit ``nn.Linear`` initialization. The official
baseline creates the output head last and then assigns its weight to the input
embedding. Creating only ``nn.Embedding`` would leave an approximately N(0, 1)
matrix and catastrophically mis-scale both input embeddings and output logits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

ActivationName = Literal["gelu", "relu2"]
ProjectionName = Literal["linear", "gated-silu"]
MixerName = Literal["attention", "shortconv"]


def rmsnorm(x0: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMSNorm matching the public NoCap baseline."""
    x = x0.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return x.to(dtype=x0.dtype)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary head dimension must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        if self.seq_len_cached != seq_len or self.cos_cached is None:
            self.seq_len_cached = seq_len
            # Match the public baseline: construct and retain RoPE in FP32.
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,T,H,D], got shape {tuple(x.shape)}")
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50_257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    embedding_dim: int = 512
    mlp_ratio: int = 4
    activation: ActivationName = "relu2"
    embedding_projection: ProjectionName = "linear"
    qk_norm: bool = False
    conv_layers: tuple[int, ...] = ()
    conv_kernel_size: int = 3

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def mixer_for_layer(self, layer_idx: int) -> MixerName:
        return "shortconv" if layer_idx in self.conv_layers else "attention"

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.head_dim % 2 != 0:
            raise ValueError("Attention head dimension must be even for RoPE")
        if self.embedding_dim <= 0 or self.embedding_dim > self.n_embd:
            raise ValueError("embedding_dim must be in [1, n_embd]")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.activation not in {"gelu", "relu2"}:
            raise ValueError(f"Unknown activation: {self.activation}")
        if self.embedding_projection not in {"linear", "gated-silu"}:
            raise ValueError(
                f"Unknown embedding projection: {self.embedding_projection}"
            )
        if self.conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if len(set(self.conv_layers)) != len(self.conv_layers):
            raise ValueError("conv_layers contains duplicate indices")
        invalid = [i for i in self.conv_layers if i < 0 or i >= self.n_layer]
        if invalid:
            raise ValueError(f"conv layer indices outside [0, {self.n_layer}): {invalid}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def config_from_preset(
    preset: str,
    *,
    embedding_dim: int | None = None,
    activation: ActivationName | None = None,
    embedding_projection: ProjectionName | None = None,
    qk_norm: bool | None = None,
    conv_layers: tuple[int, ...] | None = None,
    conv_kernel_size: int | None = None,
) -> GPTConfig:
    # LiquidLite uses C-A-A repeated four times: 4 local gated convolutions and
    # 8 unchanged global attention layers. Layer indices are zero-based here.
    liquid_layers = (0, 3, 6, 9)
    presets: dict[str, GPTConfig] = {
        "baseline": GPTConfig(
            embedding_dim=768,
            activation="gelu",
            embedding_projection="linear",
            qk_norm=False,
        ),
        "dense512": GPTConfig(
            embedding_dim=512,
            activation="relu2",
            embedding_projection="linear",
            qk_norm=False,
        ),
        "dense512-gated": GPTConfig(
            embedding_dim=512,
            activation="relu2",
            embedding_projection="gated-silu",
            qk_norm=False,
        ),
        "liquidlite512": GPTConfig(
            embedding_dim=512,
            activation="relu2",
            embedding_projection="linear",
            qk_norm=False,
            conv_layers=liquid_layers,
            conv_kernel_size=3,
        ),
        "liquidlite512-gelu": GPTConfig(
            embedding_dim=512,
            activation="gelu",
            embedding_projection="linear",
            qk_norm=False,
            conv_layers=liquid_layers,
            conv_kernel_size=3,
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(presets)}")
    base = presets[preset]
    config = GPTConfig(
        vocab_size=base.vocab_size,
        n_layer=base.n_layer,
        n_head=base.n_head,
        n_embd=base.n_embd,
        embedding_dim=embedding_dim if embedding_dim is not None else base.embedding_dim,
        mlp_ratio=base.mlp_ratio,
        activation=activation if activation is not None else base.activation,
        embedding_projection=(
            embedding_projection
            if embedding_projection is not None
            else base.embedding_projection
        ),
        qk_norm=qk_norm if qk_norm is not None else base.qk_norm,
        conv_layers=conv_layers if conv_layers is not None else base.conv_layers,
        conv_kernel_size=(
            conv_kernel_size
            if conv_kernel_size is not None
            else base.conv_kernel_size
        ),
    )
    config.validate()
    return config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.qk_norm = config.qk_norm
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim)
        k = k.view(batch, seq_len, self.n_head, self.head_dim)
        v = v.view(batch, seq_len, self.n_head, self.head_dim)

        if self.qk_norm:
            q = rmsnorm(q)
            k = rmsnorm(k)

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.c_proj(y)


class GatedShortConv(nn.Module):
    """LFM2-inspired gated causal depthwise convolution.

    This follows the public LFM2 operator shape:

        B, C, value = in_proj(x).chunk(3)
        value = B * value
        value = causal_depthwise_conv(value, kernel=3)
        value = C * value
        return out_proj(value)

    It intentionally uses only stock PyTorch operations so it works on both an
    RTX 3090 and a V100 without a custom CUDA extension. That is robust, but a
    specialized causal-conv kernel may be faster and should be benchmarked later.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        dim = config.n_embd
        self.kernel_size = config.conv_kernel_size
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.kernel_size,
            groups=dim,
            bias=False,
            padding=0,
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_b, gate_c, value = self.in_proj(x).chunk(3, dim=-1)
        value = (gate_b * value).transpose(1, 2)
        # Left-only padding makes the convolution strictly causal. Output t can
        # depend only on positions <= t. The resulting length equals input T.
        value = F.pad(value, (self.kernel_size - 1, 0))
        value = self.conv(value).transpose(1, 2)
        return self.out_proj(gate_c * value)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.activation = config.activation
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "relu2":
            x = F.relu(x).square()
        else:  # protected by GPTConfig.validate()
            raise RuntimeError(f"Unsupported activation {self.activation}")
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        mixer_kind = config.mixer_for_layer(layer_idx)
        self.mixer_kind: MixerName = mixer_kind
        self.mixer: nn.Module
        if mixer_kind == "attention":
            self.mixer = CausalSelfAttention(config)
        else:
            self.mixer = GatedShortConv(config)
        self.mlp = MLP(config)
        # Keep the proven NoCap residual scaling for both mixer types. This is a
        # deliberate stability adaptation rather than an exact copy of LFM2.
        self.mixer_scale = 1.0 / math.sqrt(2.0 * config.n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer_scale * self.mixer(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


class FactorizedVocabularyInterface(nn.Module):
    """Tied embedding/output matrix with a smaller vocabulary-facing width."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.model_dim = config.n_embd
        self.projection_kind = config.embedding_projection
        self.wte = nn.Embedding(config.vocab_size, config.embedding_dim)

        if config.embedding_dim == config.n_embd:
            self.in_proj: nn.Module = nn.Identity()
            self.in_gate: nn.Module | None = None
            self.out_proj: nn.Module = nn.Identity()
        else:
            self.in_proj = nn.Linear(config.embedding_dim, config.n_embd, bias=False)
            self.in_gate = (
                nn.Linear(config.embedding_dim, config.n_embd, bias=False)
                if config.embedding_projection == "gated-silu"
                else None
            )
            self.out_proj = nn.Linear(config.n_embd, config.embedding_dim, bias=False)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.wte(idx)
        value = self.in_proj(x)
        if self.in_gate is not None:
            value = value * F.silu(self.in_gate(x))
        return value

    def project_to_vocab(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab = FactorizedVocabularyInterface(config)
        self.blocks = nn.ModuleList(
            [Block(config, layer_idx=i) for i in range(config.n_layer)]
        )

        # Creation order is deliberate. The official model creates the Linear
        # head after all blocks, then lets that smaller initialization win.
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.vocab.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = self.vocab.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = rmsnorm(x)
        x = self.vocab.project_to_vocab(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
            )

        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizer(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        # Match the public baseline: one AdamW group for every parameter.
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def tied_weight_std(self) -> float:
        return float(self.lm_head.weight.detach().float().std().item())

    def expected_tied_weight_std(self) -> float:
        return 1.0 / math.sqrt(3.0 * self.config.embedding_dim)

    def mixer_summary(self) -> str:
        names = [block.mixer_kind for block in self.blocks]
        attention = names.count("attention")
        shortconv = names.count("shortconv")
        if shortconv == 0:
            return f"{attention} attention"
        return (
            f"{attention} attention + {shortconv} gated shortconv "
            f"(layers {[i + 1 for i in self.config.conv_layers]})"
        )
