"""LiquidLite model variant with fine-grained shared/routed MoE MLP layers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from model import (
    ActivationName,
    CausalSelfAttention,
    FactorizedVocabularyInterface,
    GPT,
    GPTConfig,
    GatedShortConv,
    MLP,
    MixerName,
    ProjectionName,
    rmsnorm,
)
from moe import FineGrainedMoE

MOE_PRESETS: Final[frozenset[str]] = frozenset({"liquidlite512-moe"})


@dataclass(frozen=True)
class MoEGPTConfig(GPTConfig):
    moe_layers: tuple[int, ...] = (2, 5, 8, 11)
    moe_num_routed_experts: int = 7
    moe_top_k: int = 4
    moe_expert_dim: int = 384
    moe_shared_expert_dim: int = 384
    moe_capacity_factor: float = 1.10
    moe_balance_loss_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001

    def validate(self) -> None:
        super().validate()
        if not self.moe_layers:
            raise ValueError("MoE architecture requires at least one MoE layer")
        if len(set(self.moe_layers)) != len(self.moe_layers):
            raise ValueError("moe_layers contains duplicate indices")
        invalid = [
            index for index in self.moe_layers if index < 0 or index >= self.n_layer
        ]
        if invalid:
            raise ValueError(
                f"MoE layer indices outside [0, {self.n_layer}): {invalid}"
            )
        if self.moe_num_routed_experts < 2:
            raise ValueError("moe_num_routed_experts must be at least 2")
        if not 1 <= self.moe_top_k <= self.moe_num_routed_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_routed_experts]")
        if self.moe_expert_dim <= 0 or self.moe_shared_expert_dim <= 0:
            raise ValueError("MoE expert dimensions must be positive")
        if self.moe_capacity_factor < 1.0:
            raise ValueError("moe_capacity_factor must be at least 1.0")
        if self.moe_balance_loss_weight < 0 or self.moe_router_z_loss_weight < 0:
            raise ValueError("MoE auxiliary-loss weights cannot be negative")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        if self.n_kv_head == self.n_head:
            values.pop("n_kv_head")
        return values


def config_from_moe_preset(
    preset: str,
    *,
    embedding_dim: int | None = None,
    activation: ActivationName | None = None,
    embedding_projection: ProjectionName | None = None,
    qk_norm: bool | None = None,
    n_kv_head: int | None = None,
    conv_layers: tuple[int, ...] | None = None,
    conv_kernel_size: int | None = None,
) -> MoEGPTConfig:
    if preset not in MOE_PRESETS:
        raise ValueError(f"Unknown MoE preset {preset!r}; choose from {sorted(MOE_PRESETS)}")
    base = MoEGPTConfig(
        n_kv_head=12,
        embedding_dim=512,
        activation="relu2",
        embedding_projection="linear",
        qk_norm=False,
        conv_layers=(0, 3, 6, 9),
        conv_kernel_size=3,
        moe_layers=(2, 5, 8, 11),
        moe_num_routed_experts=7,
        moe_top_k=4,
        moe_expert_dim=384,
        moe_shared_expert_dim=384,
        moe_capacity_factor=1.10,
        moe_balance_loss_weight=0.01,
        moe_router_z_loss_weight=0.001,
    )
    config = MoEGPTConfig(
        vocab_size=base.vocab_size,
        n_layer=base.n_layer,
        n_head=base.n_head,
        n_kv_head=n_kv_head if n_kv_head is not None else base.n_kv_head,
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
        moe_layers=base.moe_layers,
        moe_num_routed_experts=base.moe_num_routed_experts,
        moe_top_k=base.moe_top_k,
        moe_expert_dim=base.moe_expert_dim,
        moe_shared_expert_dim=base.moe_shared_expert_dim,
        moe_capacity_factor=base.moe_capacity_factor,
        moe_balance_loss_weight=base.moe_balance_loss_weight,
        moe_router_z_loss_weight=base.moe_router_z_loss_weight,
    )
    config.validate()
    return config


class MoEBlock(nn.Module):
    def __init__(self, config: MoEGPTConfig, layer_idx: int) -> None:
        super().__init__()
        mixer_kind = config.mixer_for_layer(layer_idx)
        self.mixer_kind: MixerName = mixer_kind
        self.mixer: nn.Module
        if mixer_kind == "attention":
            self.mixer = CausalSelfAttention(config)
        else:
            self.mixer = GatedShortConv(config)

        self.is_moe = layer_idx in config.moe_layers
        if self.is_moe:
            self.mlp: nn.Module = FineGrainedMoE(
                model_dim=config.n_embd,
                activation=config.activation,
                num_routed_experts=config.moe_num_routed_experts,
                top_k=config.moe_top_k,
                expert_dim=config.moe_expert_dim,
                shared_expert_dim=config.moe_shared_expert_dim,
                capacity_factor=config.moe_capacity_factor,
                balance_loss_weight=config.moe_balance_loss_weight,
                router_z_loss_weight=config.moe_router_z_loss_weight,
            )
        else:
            self.mlp = MLP(config)
        self.mixer_scale = 1.0 / math.sqrt(2.0 * config.n_layer)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.mixer_scale * self.mixer(rmsnorm(x))
        if self.is_moe:
            mlp_output, aux_loss = self.mlp(rmsnorm(x))
        else:
            mlp_output = self.mlp(rmsnorm(x))
            aux_loss = x.new_zeros(())
        return x + mlp_output, aux_loss


class MoEGPT(GPT):
    """GPT-compatible model whose selected MLP layers use Micro-MoE."""

    def __init__(self, config: MoEGPTConfig) -> None:
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        self.vocab = FactorizedVocabularyInterface(config)
        self.blocks = nn.ModuleList(
            [MoEBlock(config, layer_idx=index) for index in range(config.n_layer)]
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.vocab.wte.weight = self.lm_head.weight
        self.register_buffer(
            "last_cross_entropy", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "last_moe_aux_loss", torch.zeros((), dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = self.vocab.embed(idx)
        moe_aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x)
            moe_aux = moe_aux + block_aux
        x = rmsnorm(x)
        x = self.vocab.project_to_vocab(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            cross_entropy = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
            )
            moe_aux = moe_aux / len(self.config.moe_layers)
            if self.training:
                # Report pure CE while retaining router auxiliary gradients.
                loss = cross_entropy + moe_aux - moe_aux.detach()
            else:
                loss = cross_entropy
            self.last_cross_entropy.copy_(cross_entropy.detach().float())
            self.last_moe_aux_loss.copy_(moe_aux.detach().float())

        if not return_logits:
            logits = None
        return logits, loss

    def moe_status(self) -> dict[str, object]:
        modules = [
            block.mlp
            for block in self.blocks
            if block.is_moe and isinstance(block.mlp, FineGrainedMoE)
        ]
        loads = torch.stack([module.last_load for module in modules]).mean(dim=0)
        mean_load = loads.mean().clamp_min(1e-9)
        load_cv = loads.std(unbiased=False) / mean_load
        return {
            "layers": len(modules),
            "active_experts": modules[0].active_experts_per_token,
            "total_experts": modules[0].total_experts,
            "top_k": modules[0].top_k,
            "routed_experts": modules[0].num_routed_experts,
            "drop_rate": float(
                torch.stack([module.last_drop_rate for module in modules]).mean().item()
            ),
            "router_entropy": float(
                torch.stack(
                    [module.last_router_entropy for module in modules]
                ).mean().item()
            ),
            "aux_loss": float(self.last_moe_aux_loss.item()),
            "load_min": float(loads.min().item()),
            "load_max": float(loads.max().item()),
            "load_cv": float(load_cv.item()),
            "loads": [float(value) for value in loads.tolist()],
        }

    def mixer_summary(self) -> str:
        attention = sum(block.mixer_kind == "attention" for block in self.blocks)
        shortconv = len(self.blocks) - attention
        summary = (
            f"{attention} attention + {shortconv} gated shortconv "
            f"(layers {[index + 1 for index in self.config.conv_layers]})"
        )
        if self.config.n_kv_head != self.config.n_head:
            summary += (
                f"; GQA {self.config.n_head} query / "
                f"{self.config.n_kv_head} KV heads"
            )
        summary += (
            f"; MoE layers {[index + 1 for index in self.config.moe_layers]}: "
            f"1 shared + top-{self.config.moe_top_k}/"
            f"{self.config.moe_num_routed_experts} routed experts"
        )
        return summary
