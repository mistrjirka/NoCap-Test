"""Fine-grained shared-expert MoE used by the LiquidLite experiment."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

ActivationName = Literal["gelu", "relu2"]


def _activate(x: torch.Tensor, activation: ActivationName) -> torch.Tensor:
    if activation == "gelu":
        return F.gelu(x)
    if activation == "relu2":
        return F.relu(x).square()
    raise RuntimeError(f"Unsupported activation {activation}")


class ExpertMLP(nn.Module):
    """One always-active expert with ordinary dense Linear modules."""

    def __init__(self, model_dim: int, expert_dim: int, activation: ActivationName) -> None:
        super().__init__()
        self.activation = activation
        self.c_fc = nn.Linear(model_dim, expert_dim, bias=False)
        self.c_proj = nn.Linear(expert_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(_activate(self.c_fc(x), self.activation))


class FineGrainedMoE(nn.Module):
    """Parameter-neutral shared + top-k fine-grained MoE.

    The routed experts use a fixed-capacity batched-matrix dispatch. Capacity is
    allocated independently inside each sequence and in causal token order, so
    future tokens and other batch sequences cannot change an earlier token's route.
    Dropped routing probability is renormalized per token; the shared expert is
    always active, so every token retains a dense fallback path.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        activation: ActivationName,
        num_routed_experts: int,
        top_k: int,
        expert_dim: int,
        shared_expert_dim: int,
        capacity_factor: float,
        balance_loss_weight: float,
        router_z_loss_weight: float,
    ) -> None:
        super().__init__()
        if num_routed_experts < 2:
            raise ValueError("num_routed_experts must be at least 2")
        if not 1 <= top_k <= num_routed_experts:
            raise ValueError("top_k must be in [1, num_routed_experts]")
        if expert_dim <= 0 or shared_expert_dim <= 0:
            raise ValueError("expert dimensions must be positive")
        if capacity_factor < 1.0:
            raise ValueError("capacity_factor must be at least 1.0")
        if balance_loss_weight < 0 or router_z_loss_weight < 0:
            raise ValueError("router auxiliary weights cannot be negative")

        self.model_dim = model_dim
        self.activation = activation
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.shared_expert_dim = shared_expert_dim
        self.capacity_factor = capacity_factor
        self.balance_loss_weight = balance_loss_weight
        self.router_z_loss_weight = router_z_loss_weight

        self.shared_expert = ExpertMLP(model_dim, shared_expert_dim, activation)
        self.router = nn.Linear(model_dim, num_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # Transposed Linear-style weights allow one batched GEMM for all experts:
        # [E, capacity, D] @ [E, D, H] @ [E, H, D].
        self.expert_w1 = nn.Parameter(
            torch.empty(num_routed_experts, model_dim, expert_dim)
        )
        self.expert_w2 = nn.Parameter(
            torch.empty(num_routed_experts, expert_dim, model_dim)
        )
        nn.init.uniform_(
            self.expert_w1,
            -1.0 / math.sqrt(model_dim),
            1.0 / math.sqrt(model_dim),
        )
        nn.init.uniform_(
            self.expert_w2,
            -1.0 / math.sqrt(expert_dim),
            1.0 / math.sqrt(expert_dim),
        )

        self.register_buffer(
            "last_load",
            torch.zeros(num_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "last_drop_rate", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "last_router_entropy",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "last_aux_loss", torch.zeros((), dtype=torch.float32), persistent=False
        )

    @property
    def active_experts_per_token(self) -> int:
        return 1 + self.top_k

    @property
    def total_experts(self) -> int:
        return 1 + self.num_routed_experts


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"MoE expects [B,T,D], got {tuple(x.shape)}")
        batch, sequence_length, _ = x.shape
        num_tokens = batch * sequence_length

        # Router probabilities stay in FP32 even under BF16/FP16 autocast.
        router_logits = F.linear(x.float(), self.router.weight.float())
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(
            router_probs, self.top_k, dim=-1, sorted=False
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-9
        )

        # Assign fixed slots independently inside each sequence. Slots are allocated
        # in token order, so a future token can never change an earlier token's route.
        assignments = top_indices.reshape(batch, sequence_length * self.top_k)
        one_hot = F.one_hot(
            assignments, num_classes=self.num_routed_experts
        )
        cumulative = one_hot.cumsum(dim=1) - 1
        positions = (cumulative * one_hot).sum(dim=-1)
        capacity = min(
            sequence_length,
            max(
                1,
                math.ceil(
                    sequence_length
                    * self.top_k
                    * self.capacity_factor
                    / self.num_routed_experts
                ),
            ),
        )
        keep = positions < capacity

        token_ids = torch.arange(
            sequence_length, device=x.device
        ).repeat_interleave(self.top_k)
        token_ids = token_ids.unsqueeze(0).expand(batch, -1)
        batch_ids = torch.arange(batch, device=x.device).unsqueeze(1)
        batch_ids = batch_ids.expand_as(assignments)
        assignment_weights = top_weights.reshape(batch, -1)
        kept_weights = assignment_weights * keep.to(assignment_weights.dtype)
        safe_positions = positions.clamp(max=capacity - 1)

        # Fixed-shape dispatch avoids dynamic boolean-index outputs, which makes
        # the operator substantially friendlier to torch.compile. Dropped routes
        # contribute zero even though their clamped indices may collide.
        dispatch_index = (
            (batch_ids * self.num_routed_experts + assignments) * capacity
            + safe_positions
        )
        dispatch_source = x[batch_ids, token_ids] * keep.unsqueeze(-1).to(x.dtype)
        dispatch_flat = x.new_zeros(
            batch * self.num_routed_experts * capacity, self.model_dim
        )
        dispatch_flat.index_add_(
            0, dispatch_index.reshape(-1), dispatch_source.reshape(-1, self.model_dim)
        )
        dispatch = dispatch_flat.view(
            batch, self.num_routed_experts, capacity, self.model_dim
        )

        # ``matmul`` broadcasts expert weights across the batch without physically
        # copying them. Shapes are [B,E,C,D] @ [E,D,H] @ [E,H,D].
        hidden = torch.matmul(dispatch, self.expert_w1)
        hidden = _activate(hidden, self.activation)
        expert_outputs = torch.matmul(hidden, self.expert_w2)

        selected_outputs = expert_outputs[
            batch_ids, assignments, safe_positions
        ]
        kept_mass = router_probs.new_zeros(batch, sequence_length)
        kept_mass.scatter_add_(1, token_ids, kept_weights)
        normalized_weight = kept_weights / kept_mass.gather(
            1, token_ids
        ).clamp_min(1e-9)
        contributions = selected_outputs * normalized_weight.to(x.dtype).unsqueeze(-1)
        routed = torch.zeros_like(x)
        routed.scatter_add_(
            1,
            token_ids.unsqueeze(-1).expand(-1, -1, self.model_dim),
            contributions,
        )
        shared = self.shared_expert(x)

        # A uniform top-k mixture has approximately 1/k the variance of one expert.
        # sqrt(k) restores that variance; sqrt(2) then balances shared and routed paths.
        output = (shared + math.sqrt(self.top_k) * routed) / math.sqrt(2.0)

        assignment_fraction = one_hot.float().mean(dim=(0, 1))
        router_entropy = -(
            router_probs * router_probs.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        drop_rate = (1.0 - kept_mass.clamp(max=1.0)).mean()

        if self.training:
            mean_probability = router_probs.mean(dim=(0, 1))
            balance_loss = self.num_routed_experts * torch.sum(
                assignment_fraction.detach() * mean_probability
            )
            z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
            aux_loss = (
                self.balance_loss_weight * balance_loss
                + self.router_z_loss_weight * z_loss
            )
        else:
            aux_loss = router_logits.new_zeros(())

        self.last_load.copy_(assignment_fraction.detach())
        self.last_drop_rate.copy_(drop_rate.detach())
        self.last_router_entropy.copy_(router_entropy.detach())
        self.last_aux_loss.copy_(aux_loss.detach())

        return output, aux_loss
