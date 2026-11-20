"""LeWorldModel-style JEPA: ViT encoder, action-conditioned predictor, and SIGReg."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.nn import functional

from encoders.frames import ACTION_BLOCK_DIMENSION
from encoders.vit import MLP, Attention, TwoViewViT, ViTConfig, initialize_weights


class JEPAConfig(BaseModel):
    """Configure the encoder, projectors, predictor, and SIGReg of the JEPA."""

    model_config = ConfigDict(extra="forbid")

    encoder: ViTConfig = Field(default_factory=ViTConfig)
    latent_dim: int = Field(default=256, ge=2)
    projector_hidden: int = Field(default=2048, ge=1)
    predictor_width: int = Field(default=384, ge=8)
    predictor_depth: int = Field(default=6, ge=1)
    predictor_heads: int = Field(default=6, ge=1)
    predictor_mlp_ratio: float = Field(default=4.0, gt=0.0)
    predictor_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    history: int = Field(default=3, ge=1)
    sigreg_weight: float = Field(default=0.1, ge=0.0, allow_inf_nan=False)
    sigreg_directions: int = Field(default=1024, ge=1)
    sigreg_knots: int = Field(default=17, ge=3)
    sigreg_t_max: float = Field(default=4.0, gt=0.0)

    @model_validator(mode="after")
    def validate_predictor(self) -> JEPAConfig:
        """Require a predictor head count that divides its width."""
        if self.predictor_width % self.predictor_heads:
            raise ValueError("predictor_width must be divisible by predictor_heads")
        return self


class Projector(nn.Module):
    """An MLP with one BatchNorm hidden layer, so SIGReg can shape its output."""

    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        """Create the hidden layer, BatchNorm, and output layer."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, values: Tensor) -> Tensor:
        """Project (..., inputs) to (..., outputs), normalizing over all leading axes."""
        leading = values.shape[:-1]
        return self.layers(values.reshape(-1, values.shape[-1])).reshape(*leading, -1)


def _modulate(values: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply an adaptive LayerNorm shift and scale."""
    return values * (1.0 + scale) + shift


class AdaLNZeroBlock(nn.Module):
    """A causal transformer block whose norms and residual gates depend on actions.

    The modulation layer is zero-initialized, so every block starts as the
    identity and actions enter training gradually (DiT's adaLN-Zero).
    """

    def __init__(
        self, width: int, heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        """Create the sublayers and the zero-initialized modulation."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = Attention(width, heads, dropout)
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = MLP(width, int(width * mlp_ratio), dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def zero_initialize(self) -> None:
        """Zero the modulation output so the block starts as the identity."""
        final = self.modulation[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("AdaLN modulation must end with a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Return updated tokens and the mean absolute residual gate."""
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=-1)
        attended = self.attention(
            _modulate(self.attention_norm(tokens), shift_a, scale_a), causal=True
        )
        tokens = tokens + gate_a * attended
        tokens = tokens + gate_m * self.mlp(
            _modulate(self.mlp_norm(tokens), shift_m, scale_m)
        )
        gate_magnitude = 0.5 * (gate_a.abs().mean() + gate_m.abs().mean())
        return tokens, gate_magnitude.detach()


class Predictor(nn.Module):
    """Predict the next latent at every position of a causal latent history."""

    def __init__(self, config: JEPAConfig) -> None:
        """Build the input map, embeddings, AdaLN-zero blocks, and output projector."""
        super().__init__()
        width = config.predictor_width
        self.history = config.history
        self.input = nn.Linear(config.latent_dim, width)
        self.time_embedding = nn.Parameter(torch.zeros(1, config.history, width))
        self.action_embedding = nn.Sequential(
            nn.Linear(ACTION_BLOCK_DIMENSION, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(
                width,
                config.predictor_heads,
                config.predictor_mlp_ratio,
                config.predictor_dropout,
            )
            for _ in range(config.predictor_depth)
        )
        self.norm = nn.LayerNorm(width)
        self.projector = Projector(width, config.projector_hidden, config.latent_dim)
        self.apply(initialize_weights)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        for block in self.blocks:
            block.zero_initialize()

    def forward(self, latents: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Map (B, T, latent) and (B, T, 20) to next-latent predictions and gate size.

        Position ``k`` sees latents 0..k and predicts latent ``k + 1`` under the
        action block applied between timesteps ``k`` and ``k + 1``.
        """
        if latents.ndim != 3 or actions.shape[:2] != latents.shape[:2]:
            raise ValueError("latents and actions must share (batch, time) axes")
        if latents.shape[1] > self.history:
            raise ValueError(f"The predictor accepts at most {self.history} timesteps")
        tokens = self.input(latents) + self.time_embedding[:, : latents.shape[1]]
        condition = self.action_embedding(actions)
        gates = []
        for block in self.blocks:
            tokens, gate = block(tokens, condition)
            gates.append(gate)
        return self.projector(self.norm(tokens)), torch.stack(gates).mean()


def sigreg(
    latents: Tensor,
    directions: Tensor,
    *,
    knots: int = 17,
    t_max: float = 4.0,
) -> Tensor:
    """Return the mean Epps-Pulley statistic of 1-D projections of ``latents``.

    For each unit direction ``u``, the statistic is ``N`` times the
    Gaussian-weighted squared distance between the empirical characteristic
    function of ``latents @ u`` and that of a standard normal, integrated over
    ``t`` in ``[-t_max, t_max]`` by the trapezoidal rule (using its symmetry).
    Matching every 1-D projection matches the joint distribution (Cramer-Wold),
    so minimizing it pushes ``latents`` toward an isotropic standard normal.
    Computed in float32.
    """
    if latents.ndim != 2 or directions.ndim != 2:
        raise ValueError("latents must be (N, D) and directions (D, M)")
    if latents.shape[1] != directions.shape[0]:
        raise ValueError("directions must match the latent dimension")
    device_type = latents.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        values = latents.float()
        projected = values @ directions.float()
        t = torch.linspace(0.0, t_max, knots, device=values.device)
        arguments = projected.unsqueeze(-1) * t
        real = torch.cos(arguments).mean(dim=0)
        imaginary = torch.sin(arguments).mean(dim=0)
        normal = torch.exp(-0.5 * t.square())
        distance = (real - normal).square() + imaginary.square()
        step = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * step, device=values.device)
        weights[0] = step
        weights[-1] = step
        statistic = len(values) * (distance @ (weights * normal))
        return statistic.mean()


def random_directions(
    dimension: int,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``count`` unit directions in R^dimension as columns of a matrix."""
    directions = torch.randn(dimension, count, device=device, generator=generator)
    return directions / directions.norm(dim=0, keepdim=True)


def sigreg_null_expectation(t_max: float = 4.0) -> float:
    """Return the expected statistic for standard-normal samples (no quadrature)."""
    # E|ecf - cf|^2 = (1 - exp(-t^2)) / N, weighted by exp(-t^2 / 2).
    return float(
        math.sqrt(2.0 * math.pi) * math.erf(t_max / math.sqrt(2.0))
        - math.sqrt(2.0 * math.pi / 3.0) * math.erf(t_max * math.sqrt(1.5))
    )


class JEPALosses(NamedTuple):
    """Hold the training objective, its terms, and tensors for diagnostics."""

    total: Tensor
    prediction: Tensor
    sigreg: Tensor
    latents: Tensor
    predictions: Tensor
    gate_magnitude: Tensor


class JEPA(nn.Module):
    """Encode two-camera windows to latents and predict each next latent."""

    def __init__(self, config: JEPAConfig | None = None) -> None:
        """Build the encoder, its projector, and the predictor."""
        super().__init__()
        self.config = config or JEPAConfig()
        self.encoder = TwoViewViT(self.config.encoder)
        self.projector = Projector(
            2 * self.encoder.width, self.config.projector_hidden, self.config.latent_dim
        )
        self.projector.apply(initialize_weights)
        self.predictor = Predictor(self.config)

    def encode(self, front: Tensor, wrist: Tensor) -> Tensor:
        """Encode uint8 images (B, T, H, W, 3) to latents (B, T, latent)."""
        batch, steps = front.shape[:2]
        output = self.encoder(front.flatten(0, 1), wrist.flatten(0, 1))
        latents = self.projector(output.cls.flatten(1))
        return latents.reshape(batch, steps, -1)

    def losses(
        self,
        front: Tensor,
        wrist: Tensor,
        actions: Tensor,
        directions: Tensor,
    ) -> JEPALosses:
        """Compute next-embedding MSE plus weighted SIGReg for a batch of windows."""
        if front.shape[1] != self.config.history + 1:
            raise ValueError(
                f"Training windows must have {self.config.history + 1} timesteps"
            )
        latents = self.encode(front, wrist)
        predictions, gate_magnitude = self.predictor(latents[:, :-1], actions)
        prediction_loss = functional.mse_loss(
            predictions.float(), latents[:, 1:].float()
        )
        regularizer = torch.stack(
            [
                sigreg(
                    latents[:, step],
                    directions,
                    knots=self.config.sigreg_knots,
                    t_max=self.config.sigreg_t_max,
                )
                for step in range(latents.shape[1])
            ]
        ).mean()
        total = prediction_loss + self.config.sigreg_weight * regularizer
        return JEPALosses(
            total, prediction_loss, regularizer, latents, predictions, gate_magnitude
        )

    @torch.no_grad()
    def rollout(self, context: Tensor, actions: Tensor) -> Tensor:
        """Predict ``actions.shape[1] - context.shape[1] + 1`` future latents.

        ``context`` is (B, history, latent) and ``actions`` holds every action
        block from the first context timestep onward. Each step keeps the most
        recent ``history`` latents, real or predicted.
        """
        history = context.shape[1]
        if history != self.config.history:
            raise ValueError(f"Rollouts need {self.config.history} context latents")
        horizon = actions.shape[1] - history + 1
        if horizon < 1:
            raise ValueError("Rollouts need at least one action block past the context")
        window = context
        predictions = []
        for step in range(horizon):
            predicted, _ = self.predictor(window, actions[:, step : step + history])
            next_latent = predicted[:, -1:]
            predictions.append(next_latent)
            window = torch.cat((window[:, 1:], next_latent), dim=1)
        return torch.cat(predictions, dim=1)


class SupervisedEncoder(nn.Module):
    """The JEPA's encoder and projector with a linear head on privileged labels."""

    def __init__(self, config: JEPAConfig, outputs: int) -> None:
        """Build the encoder, projector, and linear output head."""
        super().__init__()
        self.config = config
        self.encoder = TwoViewViT(config.encoder)
        self.projector = Projector(
            2 * self.encoder.width, config.projector_hidden, config.latent_dim
        )
        self.head = nn.Linear(config.latent_dim, outputs)
        self.projector.apply(initialize_weights)
        self.head.apply(initialize_weights)

    def forward(self, front: Tensor, wrist: Tensor) -> Tensor:
        """Predict standardized targets and flag logits from uint8 images."""
        output = self.encoder(front, wrist)
        return self.head(self.projector(output.cls.flatten(1)))


def effective_rank(latents: Tensor) -> float:
    """Return exp(entropy) of the normalized singular values of centered latents."""
    values = latents.detach().float().cpu()
    values = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(values)
    total = singular.sum()
    if total <= 0:
        return 0.0
    probabilities = singular / total
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def count_parameters(module: nn.Module) -> int:
    """Return the number of parameters, whether or not they are frozen."""
    return sum(parameter.numel() for parameter in module.parameters())
