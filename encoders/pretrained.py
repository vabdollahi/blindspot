"""Frozen-feature extraction for the Phase 4 encoders, including DINOv2 ViT-S/14."""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoders.frames import EpisodeRef, FrameCache, TimestepDataset
from encoders.jepa import JEPA, SupervisedEncoder
from encoders.vit import (
    IMAGE_MEAN,
    IMAGE_STD,
    EncoderOutput,
    TwoViewViT,
    global_readout,
    grid_readout,
)

DINOV2_MODEL_ID = "facebook/dinov2-small"
DINOV2_PATCH_SIZE = 14


class FeatureExtractor(Protocol):
    """Map a batch of uint8 front and wrist images to named readouts."""

    name: str

    def __call__(self, front: Tensor, wrist: Tensor) -> dict[str, Tensor]:
        """Return readouts such as ``global``, ``grid``, and ``latent``."""
        ...


class ViTFeatureExtractor:
    """Read global and grid features (and the JEPA latent) from a two-view ViT."""

    def __init__(
        self,
        name: str,
        encoder: TwoViewViT,
        jepa: JEPA | None = None,
        supervised: SupervisedEncoder | None = None,
    ) -> None:
        """Freeze the encoder; expose a JEPA's latent or a supervised model's output."""
        self.name = name
        self.encoder = encoder.eval().requires_grad_(False)
        self.jepa = jepa.eval().requires_grad_(False) if jepa is not None else None
        self.supervised = (
            supervised.eval().requires_grad_(False) if supervised is not None else None
        )

    @torch.no_grad()
    def __call__(self, front: Tensor, wrist: Tensor) -> dict[str, Tensor]:
        """Return the global and grid readouts, plus the latent or trained output."""
        output = self.encoder(front, wrist)
        readouts = {"global": global_readout(output), "grid": grid_readout(output)}
        cls = output.cls.flatten(1)
        if self.jepa is not None:
            readouts["latent"] = self.jepa.projector(cls)
        if self.supervised is not None:
            readouts["output"] = self.supervised.head(self.supervised.projector(cls))
        return readouts


class DINOv2FeatureExtractor:
    """Run frozen DINOv2 ViT-S/14 on both cameras with the same readouts as the ViT.

    The front camera is resized to the nearest multiple of 14 px (160 -> 168);
    the wrist camera's 112 px already is. DINOv2 interpolates its position
    embeddings to the resulting 12 x 12 and 8 x 8 patch grids.
    """

    def __init__(
        self, model: torch.nn.Module, revision: str, transformers_version: str
    ) -> None:
        """Wrap a loaded, frozen DINOv2 model."""
        self.name = "dinov2-vits14"
        self.model = model.eval().requires_grad_(False)
        self.revision = revision
        self.transformers_version = transformers_version

    @torch.no_grad()
    def __call__(self, front: Tensor, wrist: Tensor) -> dict[str, Tensor]:
        """Return the global and grid readouts for uint8 images."""
        tokens = [self._tokens(images) for images in (front, wrist)]
        output = EncoderOutput(
            cls=torch.stack((tokens[0][:, 0], tokens[1][:, 0]), dim=1),
            front_patches=tokens[0][:, 1:],
            wrist_patches=tokens[1][:, 1:],
        )
        return {"global": global_readout(output), "grid": grid_readout(output)}

    def _tokens(self, images: Tensor) -> Tensor:
        """Return final-layer (layer-normalized) tokens for one camera."""
        values = images.permute(0, 3, 1, 2).to(torch.float32) / 255.0
        size = round(values.shape[-1] / DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
        if size != values.shape[-1]:
            values = functional.interpolate(
                values,
                size=(size, size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )
        mean = torch.tensor(IMAGE_MEAN, device=values.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD, device=values.device).view(1, 3, 1, 1)
        output = self.model(pixel_values=(values - mean) / std)
        return output.last_hidden_state.float()


def load_dinov2(
    revision: str = "main", *, device: torch.device
) -> DINOv2FeatureExtractor:
    """Download (if needed) and load DINOv2 ViT-S/14 at an immutable revision."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "DINOv2 features require the jepa extra; run `uv sync --extra jepa`."
        ) from error
    snapshot = Path(snapshot_download(repo_id=DINOV2_MODEL_ID, revision=revision))
    resolved = snapshot.name
    if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise RuntimeError(
            f"Hugging Face Hub returned an invalid revision: {resolved!r}"
        )
    model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=False).to(device)
    return DINOv2FeatureExtractor(
        model, resolved, importlib.metadata.version("transformers")
    )


def extract_features(
    extractor: FeatureExtractor,
    caches: Sequence[FrameCache],
    episodes: Sequence[EpisodeRef],
    *,
    device: torch.device,
    batch_size: int = 256,
    readouts: Sequence[str] = ("global", "grid"),
) -> dict[str, NDArray[np.float32]]:
    """Extract readouts for every timestep of ``episodes``, in their order."""
    dataset = TimestepDataset(caches, episodes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: dict[str, list[NDArray[np.float32]]] = {name: [] for name in readouts}
    for batch in tqdm(loader, desc=f"Features {extractor.name}", unit="batch"):
        features = extractor(batch["front"].to(device), batch["wrist"].to(device))
        for name in readouts:
            value = features[name].float().cpu().numpy()
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{extractor.name} produced non-finite {name}")
            outputs[name].append(value)
    return {name: np.concatenate(values) for name, values in outputs.items()}
