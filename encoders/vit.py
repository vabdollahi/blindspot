"""A ViT-Tiny/16 shared by the front and wrist cameras, with global and grid readouts."""

from __future__ import annotations

from typing import NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.nn import functional

# ImageNet statistics, also used by DINOv2, so every encoder sees the same input scale.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
GRID_POOL_SIZE = 4


class ViTConfig(BaseModel):
    """Configure the two-camera ViT; the defaults are ViT-Tiny/16."""

    model_config = ConfigDict(extra="forbid")

    patch_size: int = Field(default=16, ge=1)
    width: int = Field(default=192, ge=8)
    depth: int = Field(default=12, ge=1)
    heads: int = Field(default=3, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    front_size: int = Field(default=160, ge=16)
    wrist_size: int = Field(default=112, ge=16)

    @model_validator(mode="after")
    def validate_shapes(self) -> ViTConfig:
        """Require whole patch grids and a head count that divides the width."""
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        for size in (self.front_size, self.wrist_size):
            if size % self.patch_size:
                raise ValueError("image sizes must be multiples of the patch size")
        return self

    def grid_side(self, camera: str) -> int:
        """Return the patch-grid side length for one camera."""
        size = self.front_size if camera == "front" else self.wrist_size
        return size // self.patch_size


class EncoderOutput(NamedTuple):
    """Hold final-layer CLS tokens and patch tokens for both cameras."""

    cls: Tensor
    front_patches: Tensor
    wrist_patches: Tensor


class Attention(nn.Module):
    """Multi-head self-attention with an optional causal mask."""

    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        """Create the fused query/key/value and output projections."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, tokens: Tensor, *, causal: bool = False) -> Tensor:
        """Attend over the token axis of a (batch, tokens, width) tensor."""
        batch, length, width = tokens.shape
        query, key, value = (
            self.qkv(tokens)
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    """A two-layer GELU feed-forward network."""

    def __init__(self, width: int, hidden: int, dropout: float = 0.0) -> None:
        """Create the expansion and contraction layers."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply the feed-forward network to every token."""
        return self.layers(tokens)


class Block(nn.Module):
    """A pre-norm transformer block."""

    def __init__(self, width: int, heads: int, mlp_ratio: float) -> None:
        """Create the attention and feed-forward sublayers."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = MLP(width, int(width * mlp_ratio))

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply residual attention and feed-forward updates."""
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.mlp(self.mlp_norm(tokens))


def prepare_images(images: Tensor) -> Tensor:
    """Convert uint8 (B, H, W, 3) images to normalized float (B, 3, H, W)."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Images must be uint8 with shape (batch, height, width, 3)")
    values = images.permute(0, 3, 1, 2).to(torch.float32) / 255.0
    mean = torch.tensor(IMAGE_MEAN, device=values.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=values.device).view(1, 3, 1, 1)
    return (values - mean) / std


class TwoViewViT(nn.Module):
    """Encode front and wrist images with shared weights and per-camera positions.

    Each camera has its own position-embedding table (its grid size differs),
    which also tells the shared transformer which view it is encoding.
    """

    def __init__(self, config: ViTConfig | None = None) -> None:
        """Build the patch embedding, per-camera positions, and transformer."""
        super().__init__()
        self.config = config or ViTConfig()
        width = self.config.width
        self.patch_embedding = nn.Conv2d(
            3, width, self.config.patch_size, stride=self.config.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        # Each camera gets learned offsets for its CLS token and every patch;
        # adding them to the tokens injects camera-specific spatial information.
        self.position_embeddings = nn.ParameterDict(
            {
                camera: nn.Parameter(
                    torch.zeros(1, 1 + self.config.grid_side(camera) ** 2, width)
                )
                for camera in ("front", "wrist")
            }
        )
        self.blocks = nn.ModuleList(
            Block(width, self.config.heads, self.config.mlp_ratio)
            for _ in range(self.config.depth)
        )
        self.norm = nn.LayerNorm(width)
        self.apply(initialize_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for embedding in self.position_embeddings.values():
            nn.init.trunc_normal_(embedding, std=0.02)

    @property
    def width(self) -> int:
        """Return the token width."""
        return self.config.width

    def encode_camera(self, images: Tensor, camera: str) -> Tensor:
        """Return final-layer tokens (B, 1 + patches, width) for uint8 images."""
        expected = (
            self.config.front_size if camera == "front" else self.config.wrist_size
        )
        if tuple(images.shape[1:3]) != (expected, expected):
            raise ValueError(
                f"{camera} images must be {expected} x {expected}, got "
                f"{tuple(images.shape[1:3])}"
            )
        patches = (
            self.patch_embedding(prepare_images(images)).flatten(2).transpose(1, 2)
        )
        tokens = torch.cat(
            (self.cls_token.expand(len(patches), -1, -1), patches), dim=1
        )
        tokens = tokens + self.position_embeddings[camera]
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, front: Tensor, wrist: Tensor) -> EncoderOutput:
        """Encode both cameras for a batch of uint8 images."""
        front_tokens = self.encode_camera(front, "front")
        wrist_tokens = self.encode_camera(wrist, "wrist")
        return EncoderOutput(
            cls=torch.stack((front_tokens[:, 0], wrist_tokens[:, 0]), dim=1),
            front_patches=front_tokens[:, 1:],
            wrist_patches=wrist_tokens[:, 1:],
        )


def global_readout(output: EncoderOutput) -> Tensor:
    """Concatenate each camera's CLS token and mean patch token."""
    return torch.cat(
        (
            output.cls[:, 0],
            output.front_patches.mean(dim=1),
            output.cls[:, 1],
            output.wrist_patches.mean(dim=1),
        ),
        dim=-1,
    )


def grid_readout(output: EncoderOutput, pool: int = GRID_POOL_SIZE) -> Tensor:
    """Average-pool each camera's patch grid to pool x pool and concatenate."""
    pooled = []
    for patches in (output.front_patches, output.wrist_patches):
        batch, count, width = patches.shape
        side = round(count**0.5)
        if side * side != count:
            raise ValueError("Patch tokens must form a square grid")
        grid = patches.transpose(1, 2).reshape(batch, width, side, side)
        # MPS cannot pool grids that pool does not divide (10 x 10, 7 x 7), so pool on CPU.
        pooled.append(
            functional.adaptive_avg_pool2d(grid.float().cpu(), pool)
            .flatten(1)
            .to(patches.device)
        )
    return torch.cat(pooled, dim=-1)


def initialize_weights(module: nn.Module) -> None:
    """Use truncated-normal linear weights and zero biases, as in ViT."""
    if isinstance(module, nn.Linear | nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
