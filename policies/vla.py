"""Frozen-backbone SmolVLM2 VLA baseline and its shared policy adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common.training import (
    TrainingRun,
    optimizer_step,
    resolve_device,
    seed_everything,
)
from motion.control import WORKSPACE_LOWER, WORKSPACE_UPPER

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TRANSFORMERS_VERSION = "5.20.2"
CHECKPOINT_FORMAT_VERSION = 4
DEFAULT_REPLAN_INTERVAL = 2
_SINGLE_STEP_CHECKPOINT_FORMAT = 3
DEFAULT_FEATURE_LAYER = 16
VLAProfile = Literal["full", "limited"]
_ACTION_LOW = np.array([*WORKSPACE_LOWER, -np.pi / 2, 0.0], dtype=np.float32)
_ACTION_HIGH = np.array([*WORKSPACE_UPPER, np.pi / 2, 1.0], dtype=np.float32)
_ACTION_NAMES = ("x", "y", "z", "yaw", "gripper_open")
_IMAGE_KEYS = ("observation.images.front", "observation.images.wrist")
_CAMERA_COUNT = len(_IMAGE_KEYS)
# observation.state holds joints (7), gripper width (1), then TCP xyz and yaw.
_TCP_POSE = slice(8, 12)
_TOKEN_PROJECTION_SIZE = 8
_CONTEXT_PROJECTION_SIZE = 32
_MLP_WIDTH = 256
_PROMPT_PREFIX = "Follow this pick-and-place instruction: "
_PROMPT_LAYOUT = "instruction-before-images"
_FEATURE_CACHE_VERSION = 3
_STATE_STD_FLOOR = 1e-6
_FEATURE_STD_FLOOR = 1e-3


class VLAConfig(BaseModel):
    """Validate settings for the frozen-backbone SmolVLM2 T3-L action head."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = MODEL_ID
    model_revision: str = "main"
    profile: VLAProfile = "full"
    feature_layer: int = Field(default=DEFAULT_FEATURE_LAYER, ge=0)
    # Actions predicted per policy query; 1 is the single-step head.
    chunk_size: int = Field(default=1, ge=1)
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=128, ge=1)
    encoding_batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-2, ge=0.0, allow_inf_nan=False)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, allow_inf_nan=False)
    num_workers: int = Field(default=0, ge=0)
    device: str = "auto"
    seed: int = Field(default=42, ge=0)
    patience: int = Field(default=10, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    gripper_loss_weight: float = Field(default=0.2, gt=0.0, le=1.0)
    max_encoding_samples: int | None = Field(default=None, ge=1)
    max_train_steps: int | None = Field(default=None, ge=1)


class VLAEpochMetrics(BaseModel):
    """Record one action-head epoch's mean train and validation losses."""

    epoch: int
    training_l1: float
    validation_l1: float
    training_loss: float
    validation_loss: float
    training_gripper_bce: float
    validation_gripper_bce: float
    training_gripper_accuracy: float
    validation_gripper_accuracy: float


class VLAEncodingMetrics(BaseModel):
    """Record the frozen encoder's measured throughput and memory use."""

    encoded_samples: int
    cached_samples: int = Field(default=0, ge=0)
    encoding_seconds: float
    steps_per_second: float
    backbone_parameters: int
    device: str
    model_dtype: str
    memory_mb: float | None
    transformers_version: str
    model_revision: str
    feature_layer: int
    image_tokens_per_camera: int


class VLATrainingResult(BaseModel):
    """Summarize validation performance and the best selected head epoch."""

    device: str
    initial_validation_l1: float
    best_validation_l1: float
    best_epoch: int
    optimizer_steps: int
    history: list[VLAEpochMetrics]
    validation_action_mae: dict[str, float]
    validation_episode_l1: dict[str, float]
    encoding: VLAEncodingMetrics
    validation_gripper_accuracy: float
    validation_gripper_bce: float
    gripper_positive_weight: float
    initial_validation_loss: float
    best_validation_loss: float
    ablated_validation_l1: float
    ablated_validation_action_mae: dict[str, float]
    validation_chunk_l1: float | None = None


class VLAFeatureCache(BaseModel):
    """Locate reusable frozen-VLM features and describe each split's frames."""

    model_config = ConfigDict(extra="forbid")

    directory: Path
    split_keys: dict[Literal["train", "validation"], Any]


class VLAFeatures(NamedTuple):
    """Hold per-camera image-token grids and the instruction-context vector."""

    grid: Tensor
    context: Tensor


class VLANormalization(NamedTuple):
    """Hold training-split input statistics and residual-target statistics."""

    state_mean: Tensor
    state_std: Tensor
    grid_mean: Tensor
    grid_std: Tensor
    context_mean: Tensor
    context_std: Tensor
    delta_mean: Tensor
    delta_std: Tensor


class VLAFrameDataset(Dataset[dict[str, Any]]):
    """Expose LeRobot T3-L frames with both images and clean action labels."""

    def __init__(
        self,
        dataset: Any,
        *,
        frame_indices: Sequence[int] | None = None,
    ) -> None:
        """Wrap a LeRobot split, optionally selecting specific source frames."""
        self._dataset = dataset
        self.dataset_version = str(dataset.meta.info.codebase_version)
        self._numeric = dataset.select_columns(
            ["observation.state", "action", "episode_index"]
        ).with_format("numpy")
        self._frame_indices = (
            np.arange(len(self._numeric), dtype=np.int64)
            if frame_indices is None
            else np.asarray(frame_indices, dtype=np.int64)
        )
        if (
            self._frame_indices.ndim != 1
            or self._frame_indices.size == 0
            or np.any(self._frame_indices < 0)
            or np.any(self._frame_indices >= len(self._numeric))
            or np.unique(self._frame_indices).size != self._frame_indices.size
        ):
            raise ValueError("VLA frame selection must contain unique valid indices")

    def __len__(self) -> int:
        """Return the number of transitions in the wrapped split."""
        return len(self._frame_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one paired-image, language, state, and clean-action sample."""
        source_index = int(self._frame_indices[index])
        row = self._dataset[source_index]
        numeric = self._numeric[source_index]
        instruction = row.get("task")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"LeRobot row {source_index} has no non-empty task string")
        state = _finite_vector(numeric["observation.state"], (12,), "state")
        action = _finite_vector(numeric["action"], (5,), "action")
        episode_index = int(numeric["episode_index"])
        return {
            "front": _rgb_image(row.get(_IMAGE_KEYS[0]), _IMAGE_KEYS[0]),
            "wrist": _rgb_image(row.get(_IMAGE_KEYS[1]), _IMAGE_KEYS[1]),
            "instruction": instruction,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
        }

    def action_chunks(self, chunk_size: int) -> tuple[Tensor, Tensor]:
        """Return each selected frame's next clean actions and their validity mask."""
        columns = self._numeric[:]
        return action_chunk_labels(
            columns["action"],
            columns["episode_index"],
            self._frame_indices,
            chunk_size,
        )


def action_chunk_labels(
    actions: Any,
    episode_indices: Any,
    frame_indices: Any,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Gather each frame's next ``chunk_size`` clean actions within its episode.

    Positions past the end of an episode repeat the episode's final action and
    are marked invalid in the returned mask, so they can be excluded from the
    loss.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("VLA chunk size must be an integer")
    if chunk_size < 1:
        raise ValueError("VLA chunk size must be positive")
    all_actions = np.asarray(actions, dtype=np.float32)
    episodes = np.asarray(episode_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if (
        all_actions.ndim != 2
        or all_actions.shape[1] != 5
        or episodes.shape != (all_actions.shape[0],)
        or frames.ndim != 1
        or frames.size == 0
        or np.any(frames < 0)
        or np.any(frames >= all_actions.shape[0])
    ):
        raise ValueError("VLA action-chunk inputs have invalid shapes or indices")
    if not np.isfinite(all_actions).all():
        raise ValueError("VLA action labels must be finite")
    candidates = frames[:, None] + np.arange(chunk_size, dtype=np.int64)[None, :]
    clipped = np.minimum(candidates, all_actions.shape[0] - 1)
    same_episode = (candidates < all_actions.shape[0]) & (
        episodes[clipped] == episodes[frames][:, None]
    )
    valid = np.cumprod(same_episode, axis=1).astype(bool)
    last_valid = frames + valid.sum(axis=1) - 1
    sources = np.where(valid, clipped, last_valid[:, None])
    return (
        torch.from_numpy(np.ascontiguousarray(all_actions[sources])),
        torch.from_numpy(valid),
    )


class VLAEncoder(Protocol):
    """Describe the frozen multimodal encoder needed by the VLA trainer."""

    hidden_size: int
    feature_layer: int
    model_id: str
    model_revision: str
    transformers_version: str
    model_dtype: str
    device: torch.device
    parameter_count: int

    def encode(
        self,
        front: Sequence[np.ndarray],
        wrist: Sequence[np.ndarray],
        instructions: Sequence[str],
    ) -> VLAFeatures:
        """Encode paired images and instructions into token grids and context."""
        ...


class FrozenSmolVLMEncoder:
    """Read SmolVLM2 image-token and instruction states without training the VLM."""

    def __init__(
        self,
        backbone: nn.Module,
        processor: Any,
        *,
        model_id: str,
        model_revision: str,
        transformers_version: str,
        device: str | torch.device,
        feature_layer: int = DEFAULT_FEATURE_LAYER,
    ) -> None:
        """Freeze a loaded multimodal model and retain its resolved revision."""
        if isinstance(feature_layer, bool) or feature_layer < 0:
            raise ValueError("SmolVLM2 feature layer must be a non-negative integer")
        self.backbone = backbone
        self.processor = processor
        self.model_id = model_id
        self.model_revision = model_revision
        self.transformers_version = transformers_version
        self.feature_layer = feature_layer
        self.device = resolve_device(device)
        self.backbone.to(self.device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        text_config = getattr(self.backbone.config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("SmolVLM2 config must define text_config.hidden_size")
        layer_count = getattr(text_config, "num_hidden_layers", None)
        if isinstance(layer_count, int) and feature_layer > layer_count:
            raise ValueError(
                f"SmolVLM2 has {layer_count} decoder layers; "
                f"feature layer {feature_layer} does not exist"
            )
        image_token_id = getattr(self.processor, "image_token_id", None)
        if isinstance(image_token_id, bool) or not isinstance(image_token_id, int):
            raise TypeError("SmolVLM2 processor must expose an integer image_token_id")
        self._image_token_id = image_token_id
        self.hidden_size = hidden_size
        self.parameter_count = sum(
            parameter.numel() for parameter in self.backbone.parameters()
        )
        self._base_model = getattr(self.backbone, "model", None)
        if not isinstance(self._base_model, nn.Module):
            raise TypeError("SmolVLM2 must expose its multimodal base as model")
        parameters = iter(self.backbone.parameters())
        first_parameter = next(parameters, None)
        if first_parameter is None:
            raise ValueError("SmolVLM2 backbone has no parameters")
        self._dtype = first_parameter.dtype
        self.model_dtype = str(self._dtype).removeprefix("torch.")

    def encode(
        self,
        front: Sequence[np.ndarray],
        wrist: Sequence[np.ndarray],
        instructions: Sequence[str],
    ) -> VLAFeatures:
        """Return each camera's image-token states and the final prompt state."""
        if not front or len(front) != len(wrist) or len(front) != len(instructions):
            raise ValueError("VLA image and instruction batches must have equal size")
        if any(not instruction.strip() for instruction in instructions):
            raise ValueError("VLA instructions must be non-empty")

        # The decoder is causal: image tokens only see earlier tokens, so the
        # instruction must precede the images for their states to depend on it.
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{_PROMPT_PREFIX}{instruction}"},
                        {"type": "image"},
                        {"type": "image"},
                    ],
                }
            ]
            for instruction in instructions
        ]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not callable(
            getattr(tokenizer, "apply_chat_template", None)
        ):
            raise TypeError("SmolVLM2 processor must expose a chat-template tokenizer")
        prompts = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompts, str):
            prompts = [prompts]
        # The default processor upsamples each 256 px frame to 2048 px and
        # splits it into 17 tiles; one 512 px view per camera keeps 64 tokens.
        encoded = self.processor(
            text=prompts,
            images=[list(pair) for pair in zip(front, wrist, strict=True)],
            padding=True,
            return_tensors="pt",
            do_image_splitting=False,
        )
        model_inputs = {
            name: _move_tensor(value, self.device, self._dtype)
            for name, value in encoded.items()
            if isinstance(value, Tensor)
        }
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("SmolVLM2 processor must provide 2-D input_ids")
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "SmolVLM2 processor must provide a matching attention_mask"
            )
        with torch.inference_mode():
            output = self._base_model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = getattr(output, "hidden_states", None)
        if (
            not isinstance(hidden_states, (tuple, list))
            or len(hidden_states) <= self.feature_layer
        ):
            raise TypeError(
                f"SmolVLM2 did not return hidden states for layer {self.feature_layer}"
            )
        layer = hidden_states[self.feature_layer]
        batch_size = len(instructions)
        if not isinstance(layer, Tensor) or layer.shape[:2] != input_ids.shape:
            raise TypeError("SmolVLM2 hidden states do not match the input tokens")
        image_mask = input_ids == self._image_token_id
        image_token_counts = image_mask.sum(dim=1)
        image_tokens = int(image_token_counts[0])
        if (
            image_tokens < _CAMERA_COUNT
            or image_tokens % _CAMERA_COUNT
            or torch.any(image_token_counts != image_tokens)
        ):
            raise ValueError(
                "Each SmolVLM2 prompt must contain the same image-token count "
                "for both cameras"
            )
        grid = layer[image_mask].reshape(
            batch_size,
            _CAMERA_COUNT,
            image_tokens // _CAMERA_COUNT,
            self.hidden_size,
        )
        positions = torch.arange(
            attention_mask.shape[1], device=self.device, dtype=torch.int64
        ).expand_as(attention_mask)
        last_positions = (
            positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        )
        if torch.any(last_positions < 0):
            raise ValueError("SmolVLM2 processor returned an empty text sequence")
        batch_indices = torch.arange(batch_size, device=self.device)
        context = layer[batch_indices, last_positions]
        features = VLAFeatures(grid.to(torch.float32), context.to(torch.float32))
        if (
            not torch.isfinite(features.grid).all()
            or not torch.isfinite(features.context).all()
        ):
            raise FloatingPointError("SmolVLM2 produced non-finite features")
        return features


class VLAActionHead(nn.Module):
    """Map frozen token grids, context, and state to TCP deltas and gripper logits."""

    def __init__(
        self,
        hidden_size: int,
        image_tokens: int,
        *,
        chunk_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Create the per-token projection, MLP, and normalization buffers."""
        super().__init__()
        if hidden_size < 1 or image_tokens < 1 or chunk_size < 1:
            raise ValueError(
                "hidden_size, image_tokens, and chunk_size must be positive"
            )
        self.hidden_size = hidden_size
        self.image_tokens = image_tokens
        self.chunk_size = chunk_size
        self.register_buffer("state_mean", torch.zeros(12))
        self.register_buffer("state_std", torch.ones(12))
        self.register_buffer("grid_mean", torch.zeros(_CAMERA_COUNT, hidden_size))
        self.register_buffer("grid_std", torch.ones(_CAMERA_COUNT, hidden_size))
        self.register_buffer("context_mean", torch.zeros(hidden_size))
        self.register_buffer("context_std", torch.ones(hidden_size))
        self.register_buffer("delta_mean", torch.zeros(chunk_size, 4))
        self.register_buffer("delta_std", torch.ones(chunk_size, 4))
        self.token_projection = nn.Linear(hidden_size, _TOKEN_PROJECTION_SIZE)
        self.context_projection = nn.Linear(hidden_size, _CONTEXT_PROJECTION_SIZE)
        self.dropout = nn.Dropout(dropout)
        visual_size = (
            _CAMERA_COUNT * image_tokens * _TOKEN_PROJECTION_SIZE
            + _CONTEXT_PROJECTION_SIZE
        )
        # Outputs are chunk_size xyz/yaw deltas followed by chunk_size logits.
        self.layers = nn.Sequential(
            nn.Linear(visual_size + 12, _MLP_WIDTH),
            nn.GELU(),
            nn.Linear(_MLP_WIDTH, _MLP_WIDTH),
            nn.GELU(),
            nn.Linear(_MLP_WIDTH, 5 * chunk_size),
        )

    def set_normalization(self, statistics: VLANormalization) -> None:
        """Fit input standardization and residual-target scaling from training data."""
        expected_shapes = {
            "state_mean": (12,),
            "state_std": (12,),
            "grid_mean": (_CAMERA_COUNT, self.hidden_size),
            "grid_std": (_CAMERA_COUNT, self.hidden_size),
            "context_mean": (self.hidden_size,),
            "context_std": (self.hidden_size,),
            "delta_mean": (self.chunk_size, 4),
            "delta_std": (self.chunk_size, 4),
        }
        for name, shape in expected_shapes.items():
            value = getattr(statistics, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"VLA {name} must be finite with shape {shape}")
            if name.endswith("_std") and torch.any(value <= 0.0):
                raise ValueError(f"VLA {name} must be positive")
            getattr(self, name).copy_(value.to(getattr(self, name)))

    def normalization(self) -> VLANormalization:
        """Return copies of the head's normalization buffers on the CPU."""
        return VLANormalization(
            *(
                getattr(self, name).detach().cpu().clone()
                for name in VLANormalization._fields
            )
        )

    def forward(self, grid: Tensor, context: Tensor, state: Tensor) -> Tensor:
        """Predict normalized TCP xyz/yaw deltas and open-gripper logits."""
        batch_size = state.shape[0]
        if grid.shape != (
            batch_size,
            _CAMERA_COUNT,
            self.image_tokens,
            self.hidden_size,
        ):
            raise ValueError("VLA image-token grid has an invalid shape")
        if context.shape != (batch_size, self.hidden_size):
            raise ValueError("VLA context features have an invalid shape")
        if state.shape != (batch_size, 12):
            raise ValueError("VLA state inputs must have shape (batch, 12)")
        grid = (grid - self.grid_mean[:, None, :]) / self.grid_std[:, None, :]
        context = (context - self.context_mean) / self.context_std
        normalized_state = (state - self.state_mean) / self.state_std
        visual = torch.cat(
            (
                self.token_projection(grid).flatten(1),
                self.context_projection(context),
            ),
            dim=1,
        )
        return self.layers(torch.cat((self.dropout(visual), normalized_state), dim=1))

    def split_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        """Split raw outputs into normalized deltas (B, k, 4) and logits (B, k)."""
        batch_size = output.shape[0]
        if output.shape != (batch_size, 5 * self.chunk_size):
            raise ValueError("VLA head output has an invalid shape")
        deltas = output[:, : 4 * self.chunk_size].reshape(
            batch_size, self.chunk_size, 4
        )
        return deltas, output[:, 4 * self.chunk_size :]

    def absolute_actions(self, output: Tensor, state: Tensor) -> Tensor:
        """Convert head outputs into absolute xyz/yaw targets and gripper logits.

        Every chunk position is relative to the TCP pose at prediction time.
        """
        deltas, logits = self.split_output(output)
        targets = state[:, None, _TCP_POSE] + deltas * self.delta_std + self.delta_mean
        return torch.cat((targets, logits.unsqueeze(-1)), dim=-1)

    def delta_targets(self, state: Tensor, action_chunks: Tensor) -> Tensor:
        """Express clean absolute action chunks as normalized TCP deltas."""
        return (
            action_chunks[..., :4] - state[:, None, _TCP_POSE] - self.delta_mean
        ) / self.delta_std


class EncodedVLADataset(Dataset[dict[str, Tensor]]):
    """Keep frozen features and numeric targets in host memory for head training."""

    def __init__(
        self,
        grid: Tensor,
        context: Tensor,
        states: Tensor,
        actions: Tensor,
        episode_indices: Tensor,
        *,
        action_chunks: Tensor | None = None,
        chunk_mask: Tensor | None = None,
    ) -> None:
        """Validate and retain encoded CPU tensors for repeated head epochs."""
        count = grid.shape[0]
        if (
            count < 1
            or grid.ndim != 4
            or grid.shape[1] != _CAMERA_COUNT
            or context.shape != (count, grid.shape[3])
            or states.shape != (count, 12)
            or actions.shape != (count, 5)
            or episode_indices.shape != (count,)
        ):
            raise ValueError("Encoded VLA tensors have incompatible shapes")
        if (action_chunks is None) != (chunk_mask is None):
            raise ValueError("VLA action chunks and their mask must be given together")
        if action_chunks is None or chunk_mask is None:
            action_chunks = actions[:, None, :]
            chunk_mask = torch.ones((count, 1), dtype=torch.bool)
        if (
            action_chunks.ndim != 3
            or action_chunks.shape[0] != count
            or action_chunks.shape[2] != 5
            or chunk_mask.shape != action_chunks.shape[:2]
            or chunk_mask.dtype != torch.bool
            or not torch.all(chunk_mask[:, 0])
        ):
            raise ValueError("VLA action chunks have an invalid shape or mask")
        if not torch.equal(action_chunks[:, 0], actions):
            raise ValueError("VLA action chunks do not start at the encoded actions")
        self.grid = grid.contiguous()
        self.context = context.contiguous()
        self.states = states.contiguous()
        self.actions = actions.contiguous()
        self.episode_indices = episode_indices.contiguous()
        self.action_chunks = action_chunks.contiguous()
        self.chunk_mask = chunk_mask.contiguous()

    @property
    def image_tokens(self) -> int:
        """Return the number of image tokens per camera."""
        return int(self.grid.shape[2])

    @property
    def chunk_size(self) -> int:
        """Return the number of future actions labelled for every frame."""
        return int(self.action_chunks.shape[1])

    def with_action_chunks(
        self,
        action_chunks: Tensor,
        chunk_mask: Tensor,
    ) -> EncodedVLADataset:
        """Return the same encoded frames labelled with multi-step action chunks."""
        return EncodedVLADataset(
            self.grid,
            self.context,
            self.states,
            self.actions,
            self.episode_indices,
            action_chunks=action_chunks,
            chunk_mask=chunk_mask,
        )

    def __len__(self) -> int:
        """Return the number of encoded demonstration steps."""
        return self.grid.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one encoded action-head training example."""
        return {
            "grid": self.grid[index],
            "context": self.context[index],
            "state": self.states[index],
            "action_chunk": self.action_chunks[index],
            "chunk_mask": self.chunk_mask[index],
            "episode_index": self.episode_indices[index],
        }


class VLAActionPolicy:
    """Lazy-loading adapter for the shared evaluation Policy protocol.

    A chunked head predicts the next ``chunk_size`` actions per VLM query. The
    policy queries every ``replan_interval`` steps and executes the weighted
    average of every stored plan that covers the current step; a plan made
    ``a`` steps ago has weight ``exp(-ensemble_decay * a)``.
    """

    def __init__(
        self,
        head: VLAActionHead | None,
        encoder: VLAEncoder | None,
        *,
        model_id: str,
        model_revision: str,
        head_state: Mapping[str, Tensor],
        image_tokens: int,
        training_profile: VLAProfile = "full",
        feature_layer: int = DEFAULT_FEATURE_LAYER,
        chunk_size: int = 1,
        replan_interval: int | None = None,
        ensemble_decay: float = 0.0,
        device: str | torch.device = "auto",
    ) -> None:
        """Bind a trained head to a loaded or lazily loaded frozen VLM."""
        if not model_id.strip() or not model_revision.strip():
            raise ValueError("VLA model ID and revision must be non-empty")
        if not isinstance(training_profile, str) or training_profile not in {
            "full",
            "limited",
        }:
            raise ValueError("VLA training_profile must be 'full' or 'limited'")
        if image_tokens < 1 or feature_layer < 0 or chunk_size < 1:
            raise ValueError("VLA image tokens, feature layer, and chunk size invalid")
        resolved_interval = (
            min(DEFAULT_REPLAN_INTERVAL, chunk_size)
            if replan_interval is None
            else replan_interval
        )
        if not 1 <= resolved_interval <= chunk_size:
            raise ValueError("VLA replan_interval must be between 1 and chunk_size")
        if not np.isfinite(ensemble_decay) or ensemble_decay < 0.0:
            raise ValueError("VLA ensemble_decay must be finite and non-negative")
        self.device = resolve_device(device)
        self.model_id = model_id
        self.model_revision = model_revision
        self.training_profile = training_profile
        self.feature_layer = feature_layer
        self.image_tokens = image_tokens
        self.chunk_size = chunk_size
        self.replan_interval = resolved_interval
        self.ensemble_decay = float(ensemble_decay)
        self._head_state = {
            name: tensor.detach().cpu() for name, tensor in head_state.items()
        }
        self.head = head
        self.encoder = encoder
        if (head is None) != (encoder is None):
            raise ValueError(
                "VLA policy must receive both head and encoder, or neither"
            )
        if head is not None:
            if encoder is None or head.hidden_size != encoder.hidden_size:
                raise ValueError("VLA head and encoder hidden sizes must match")
            if head.image_tokens != image_tokens or head.chunk_size != chunk_size:
                raise ValueError("VLA head image tokens or chunk size do not match")
            self.head = head.to(self.device).eval()
            if not self._head_state:
                self._head_state = {
                    name: tensor.detach().cpu()
                    for name, tensor in self.head.state_dict().items()
                }
        self._instruction: str | None = None
        self._step = 0
        self._plans: list[_ActionPlan] = []

    def reset(self, instruction: str | None = None) -> None:
        """Set the episode's language command and discard earlier plans."""
        self._instruction = (
            instruction.strip()
            if isinstance(instruction, str) and instruction.strip()
            else None
        )
        self._step = 0
        self._plans = []

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Return a clipped 5-D action from images, state, and reset instruction."""
        instruction = self._instruction
        if instruction is None:
            raise RuntimeError("VLAActionPolicy requires an instruction at reset")
        self._plans = [
            plan for plan in self._plans if plan.start + self.chunk_size > self._step
        ]
        if self._step % self.replan_interval == 0 or not self._plans:
            self._plans.append(self._plan(obs, instruction))
        ages = np.array([self._step - plan.start for plan in self._plans])
        weights = np.exp(-self.ensemble_decay * ages)
        targets = np.stack(
            [plan.targets[age] for plan, age in zip(self._plans, ages, strict=True)]
        )
        open_probabilities = np.array(
            [
                plan.open_probabilities[age]
                for plan, age in zip(self._plans, ages, strict=True)
            ]
        )
        target = weights @ targets / weights.sum()
        open_probability = float(weights @ open_probabilities / weights.sum())
        self._step += 1
        action = np.array(
            [*target, float(open_probability >= 0.5)],
            dtype=np.float32,
        )
        return np.clip(action, _ACTION_LOW, _ACTION_HIGH)

    def _plan(self, obs: Mapping[str, Any], instruction: str) -> _ActionPlan:
        """Query the VLM once and return absolute targets for the next chunk."""
        encoder, head = self._ensure_loaded()
        state = _finite_vector(obs.get("observation.state"), (12,), "state")
        front = _rgb_image(obs.get(_IMAGE_KEYS[0]), _IMAGE_KEYS[0])
        wrist = _rgb_image(obs.get(_IMAGE_KEYS[1]), _IMAGE_KEYS[1])
        with torch.inference_mode():
            features = encoder.encode([front], [wrist], [instruction])
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            output = head(
                features.grid.to(self.device),
                features.context.to(self.device),
                state_tensor,
            )
            actions = head.absolute_actions(output, state_tensor)[0]
        plan = actions.detach().cpu().numpy().astype(np.float64)
        if plan.shape != (self.chunk_size, 5) or not np.isfinite(plan).all():
            raise FloatingPointError("VLA action head must emit finite action chunks")
        return _ActionPlan(
            start=self._step,
            targets=plan[:, :4],
            open_probabilities=0.5 * (1.0 + np.tanh(0.5 * plan[:, 4])),
        )

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the large VLM from process-pool pickles used by evaluation."""
        state = self.__dict__.copy()
        state["head"] = None
        state["encoder"] = None
        state["_instruction"] = None
        state["_step"] = 0
        state["_plans"] = []
        return state

    def _ensure_loaded(self) -> tuple[VLAEncoder, VLAActionHead]:
        """Load the frozen VLM and restore the small head on first inference."""
        if self.encoder is None:
            encoder = load_frozen_smolvlm(
                self.model_id,
                self.model_revision,
                device=self.device,
                feature_layer=self.feature_layer,
            )
            if encoder.model_revision != self.model_revision:
                raise RuntimeError("Loaded VLM revision does not match the checkpoint")
            if encoder.feature_layer != self.feature_layer:
                raise RuntimeError("Loaded VLM feature layer does not match")
            self.encoder = encoder
        if self.head is None:
            head = VLAActionHead(
                self.encoder.hidden_size,
                self.image_tokens,
                chunk_size=self.chunk_size,
            )
            head.load_state_dict(self._head_state, strict=True)
            self.head = head.to(self.device).eval()
        return self.encoder, self.head


class _ActionPlan(NamedTuple):
    """Hold one VLM query's absolute targets and open probabilities per step."""

    start: int
    targets: np.ndarray
    open_probabilities: np.ndarray


def load_frozen_smolvlm(
    model_id: str = MODEL_ID,
    revision: str = "main",
    *,
    device: str | torch.device = "auto",
    feature_layer: int = DEFAULT_FEATURE_LAYER,
) -> FrozenSmolVLMEncoder:
    """Load the pinned Transformers SmolVLM2 model and multimodal processor."""
    if not model_id.strip() or not revision.strip():
        raise ValueError("VLM model ID and revision must be non-empty")
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "VLA training requires the optional VLA dependencies; "
            "run `uv sync --extra vla`."
        ) from error

    version = importlib.metadata.version("transformers")
    if version != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"VLA requires transformers=={TRANSFORMERS_VERSION}, found {version}"
        )
    selected_device = resolve_device(device)
    dtype = _model_dtype(selected_device)
    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
        )
    )
    resolved_revision = snapshot_path.name
    if re.fullmatch(r"[0-9a-f]{40}", resolved_revision) is None:
        raise RuntimeError(
            f"Hugging Face Hub returned an invalid model snapshot revision: "
            f"{resolved_revision!r}"
        )
    backbone = AutoModelForImageTextToText.from_pretrained(
        str(snapshot_path),
        dtype=dtype,
        trust_remote_code=False,
    )
    processor = AutoProcessor.from_pretrained(
        str(snapshot_path),
        trust_remote_code=False,
    )
    return FrozenSmolVLMEncoder(
        backbone,
        processor,
        model_id=model_id,
        model_revision=resolved_revision,
        transformers_version=version,
        device=selected_device,
        feature_layer=feature_layer,
    )


def train_vla(
    train_dataset: Dataset[dict[str, Any]],
    validation_dataset: Dataset[dict[str, Any]],
    config: VLAConfig,
    *,
    encoder: VLAEncoder | None = None,
    run: TrainingRun | None = None,
    initial_head: VLAActionHead | None = None,
    optimizer_step_offset: int = 0,
    epoch_offset: int = 0,
    initial_best_epoch: int = 0,
    initial_best_loss: float | None = None,
    feature_cache: VLAFeatureCache | None = None,
) -> tuple[VLAActionHead, VLATrainingResult, VLAEncoder]:
    """Encode both splits once and train or warm-start the small action head."""
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("VLA training and validation datasets must be non-empty")
    offsets = (optimizer_step_offset, epoch_offset, initial_best_epoch)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in offsets):
        raise TypeError("VLA training offsets must be integers")
    if any(value < 0 for value in offsets):
        raise ValueError("VLA training offsets must be non-negative")
    if initial_best_epoch > epoch_offset:
        raise ValueError("Initial best epoch cannot exceed the epoch offset")
    if initial_head is None and (any(offsets) or initial_best_loss is not None):
        raise ValueError(
            "Training offsets and an initial best loss require an initial VLA head"
        )
    if initial_best_loss is not None and not np.isfinite(initial_best_loss):
        raise ValueError("Initial best VLA validation loss must be finite")
    if encoder is None:
        encoder = load_frozen_smolvlm(
            config.model_id,
            config.model_revision,
            device=config.device,
            feature_layer=config.feature_layer,
        )
    if encoder.feature_layer != config.feature_layer:
        raise ValueError("VLA encoder feature layer does not match the config")
    device = encoder.device
    if config.device != "auto" and resolve_device(config.device) != device:
        raise ValueError("VLA encoder device does not match the training config")
    _reset_peak_memory(encoder.device)
    train_chunks, train_chunk_mask = _chunk_labels(train_dataset, config)
    validation_chunks, validation_chunk_mask = _chunk_labels(validation_dataset, config)
    train_encoded, train_seconds, train_cached = _load_or_encode(
        train_dataset,
        encoder,
        config,
        "train",
        feature_cache,
        expected_actions=train_chunks[:, 0],
    )
    validation_encoded, validation_seconds, validation_cached = _load_or_encode(
        validation_dataset,
        encoder,
        config,
        "validation",
        feature_cache,
        expected_actions=validation_chunks[:, 0],
    )
    train_encoded = train_encoded.with_action_chunks(train_chunks, train_chunk_mask)
    validation_encoded = validation_encoded.with_action_chunks(
        validation_chunks, validation_chunk_mask
    )
    if validation_encoded.image_tokens != train_encoded.image_tokens:
        raise ValueError("VLA train and validation image-token counts differ")
    encoded_samples = len(train_encoded) + len(validation_encoded)
    encoding_seconds = train_seconds + validation_seconds
    if not np.isfinite(encoding_seconds) or encoding_seconds <= 0.0:
        raise RuntimeError("VLA feature encoding did not produce a valid duration")
    encoding = VLAEncodingMetrics(
        encoded_samples=encoded_samples,
        cached_samples=(
            (len(train_encoded) if train_cached else 0)
            + (len(validation_encoded) if validation_cached else 0)
        ),
        encoding_seconds=encoding_seconds,
        steps_per_second=encoded_samples / encoding_seconds,
        backbone_parameters=encoder.parameter_count,
        device=str(device),
        model_dtype=encoder.model_dtype,
        memory_mb=_device_memory_mb(device),
        transformers_version=encoder.transformers_version,
        model_revision=encoder.model_revision,
        feature_layer=encoder.feature_layer,
        image_tokens_per_camera=train_encoded.image_tokens,
    )

    statistics = _training_statistics(train_encoded)
    # Seed after encoding: data loading consumes global RNG, but cache hits do not.
    seed_everything(config.seed)
    if initial_head is None:
        head = VLAActionHead(
            encoder.hidden_size,
            train_encoded.image_tokens,
            chunk_size=config.chunk_size,
            dropout=config.dropout,
        )
        head.set_normalization(statistics)
        head = head.to(device)
    else:
        if (
            initial_head.hidden_size != encoder.hidden_size
            or initial_head.image_tokens != train_encoded.image_tokens
        ):
            raise ValueError("Warm-start VLA head does not match the encoder features")
        if initial_head.chunk_size != config.chunk_size:
            raise ValueError("Warm-start VLA head predicts a different chunk size")
        # Feature statistics stay with the head: re-encoded fp16 features can
        # differ slightly, while state and action statistics must match exactly.
        existing = initial_head.normalization()
        for name in ("state_mean", "state_std", "delta_mean", "delta_std"):
            if not torch.allclose(
                getattr(existing, name),
                getattr(statistics, name),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "Warm-start VLA head normalization does not match training data"
                )
        initial_head.dropout.p = config.dropout
        head = initial_head.to(device)
    train_gripper_targets = _binary_gripper_targets(
        train_encoded.action_chunks[..., 4][train_encoded.chunk_mask]
    )
    open_count = int(train_gripper_targets.sum().item())
    closed_count = len(train_gripper_targets) - open_count
    gripper_positive_weight = (
        closed_count / open_count if open_count > 0 and closed_count > 0 else 1.0
    )
    gripper_positive_weight_tensor = torch.tensor(
        gripper_positive_weight,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    validation_loader = DataLoader(
        validation_encoded,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    initial_validation = _evaluate_head(
        head,
        validation_loader,
        device,
        gripper_loss_weight=config.gripper_loss_weight,
        gripper_positive_weight=gripper_positive_weight_tensor,
    )
    best_validation_l1 = initial_validation.l1
    # A resumed run must beat the recorded best, not the re-encoded estimate
    # of it, so the saved head and the merged run record always agree.
    best_validation_loss = (
        initial_validation.objective_loss
        if initial_best_loss is None
        else initial_best_loss
    )
    best_epoch = initial_best_epoch
    best_state = _copy_state_dict(head)
    epochs_without_improvement = 0
    history: list[VLAEpochMetrics] = []
    optimizer_steps = optimizer_step_offset
    invocation_optimizer_steps = 0
    generator = torch.Generator().manual_seed(config.seed + optimizer_step_offset)
    training_loader = DataLoader(
        train_encoded,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    for epoch in range(1, config.epochs + 1):
        head.train()
        train_action_l1_sum = 0.0
        train_objective_loss_sum = 0.0
        train_gripper_correct = 0
        train_gripper_bce_sum = 0.0
        train_samples = 0
        for batch in tqdm(training_loader, desc=f"VLA head {epoch}", unit="batch"):
            grid, context, state, action_chunk, chunk_mask = _batch_to_device(
                batch, device
            )
            output = head(grid, context, state)
            action_loss, gripper_bce = _chunk_losses(
                head,
                output,
                state,
                action_chunk,
                chunk_mask,
                gripper_positive_weight_tensor,
            )
            loss = (
                1.0 - config.gripper_loss_weight
            ) * action_loss + config.gripper_loss_weight * gripper_bce
            action = action_chunk[:, 0]
            gripper_target = _binary_gripper_targets(action[:, 4])
            gripper_prediction = head.split_output(output)[1][:, 0] >= 0.0
            action_l1 = _action_errors(
                head.absolute_actions(output.detach(), state)[:, 0],
                action,
            ).mean()
            optimizer_steps += 1
            invocation_optimizer_steps += 1
            gradient_norm = optimizer_step(
                loss,
                head,
                optimizer,
                step=optimizer_steps,
                max_grad_norm=config.max_grad_norm,
            )
            gripper_correct = gripper_prediction == gripper_target.bool()
            if run is not None:
                run.log_metrics(
                    optimizer_steps,
                    {
                        "train/loss": loss.detach(),
                        "train/action_l1": action_l1,
                        "train/gripper_bce": gripper_bce.detach(),
                        "train/gripper_accuracy": gripper_correct.to(
                            torch.float32
                        ).mean(),
                        "train/gradient_norm": gradient_norm,
                    },
                    optimizer=optimizer,
                )
            train_action_l1_sum += float(action_l1) * len(action)
            train_objective_loss_sum += float(loss.detach()) * len(action)
            train_gripper_correct += int(gripper_correct.sum().item())
            train_gripper_bce_sum += float(gripper_bce.detach()) * len(action)
            train_samples += len(action)
            if (
                config.max_train_steps is not None
                and invocation_optimizer_steps >= config.max_train_steps
            ):
                break
        if train_samples == 0:
            raise RuntimeError("VLA action-head training processed no samples")
        validation = _evaluate_head(
            head,
            validation_loader,
            device,
            gripper_loss_weight=config.gripper_loss_weight,
            gripper_positive_weight=gripper_positive_weight_tensor,
        )
        history.append(
            VLAEpochMetrics(
                epoch=epoch_offset + epoch,
                training_l1=train_action_l1_sum / train_samples,
                validation_l1=validation.l1,
                training_loss=train_objective_loss_sum / train_samples,
                validation_loss=validation.objective_loss,
                training_gripper_bce=train_gripper_bce_sum / train_samples,
                validation_gripper_bce=validation.gripper_bce,
                training_gripper_accuracy=train_gripper_correct / train_samples,
                validation_gripper_accuracy=validation.gripper_accuracy,
            )
        )
        if run is not None:
            run.log_metrics(
                optimizer_steps,
                {
                    "validation/loss": validation.objective_loss,
                    "validation/action_l1": validation.l1,
                    "validation/gripper_bce": validation.gripper_bce,
                    "validation/gripper_accuracy": validation.gripper_accuracy,
                    **{
                        f"validation/mae_{name}": value
                        for name, value in validation.action_mae.items()
                    },
                },
            )
        if validation.objective_loss < best_validation_loss:
            best_validation_l1 = validation.l1
            best_validation_loss = validation.objective_loss
            best_epoch = epoch_offset + epoch
            best_state = _copy_state_dict(head)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            config.max_train_steps is not None
            and invocation_optimizer_steps >= config.max_train_steps
        ):
            break
        if epochs_without_improvement >= config.patience:
            break

    head.load_state_dict(best_state)
    head.eval()
    evaluation_options = {
        "gripper_loss_weight": config.gripper_loss_weight,
        "gripper_positive_weight": gripper_positive_weight_tensor,
    }
    final_validation = _evaluate_head(
        head, validation_loader, device, **evaluation_options
    )
    ablated_validation = _evaluate_head(
        head,
        validation_loader,
        device,
        ablate_features=True,
        **evaluation_options,
    )
    if run is not None:
        run.log_metrics(
            optimizer_steps,
            {
                "validation/best_action_l1": final_validation.l1,
                "validation/chunk_l1": final_validation.chunk_l1,
                "validation/ablated_action_l1": ablated_validation.l1,
            },
        )
    result = VLATrainingResult(
        device=str(device),
        initial_validation_l1=initial_validation.l1,
        best_validation_l1=best_validation_l1,
        best_epoch=best_epoch,
        optimizer_steps=optimizer_steps,
        history=history,
        validation_action_mae=final_validation.action_mae,
        validation_episode_l1=final_validation.episode_l1,
        encoding=encoding,
        validation_gripper_accuracy=final_validation.gripper_accuracy,
        validation_gripper_bce=final_validation.gripper_bce,
        gripper_positive_weight=gripper_positive_weight,
        initial_validation_loss=initial_validation.objective_loss,
        best_validation_loss=best_validation_loss,
        ablated_validation_l1=ablated_validation.l1,
        ablated_validation_action_mae=ablated_validation.action_mae,
        validation_chunk_l1=final_validation.chunk_l1,
    )
    return head, result, encoder


def save_vla_checkpoint(
    head: VLAActionHead,
    model_id: str,
    model_revision: str,
    path: str | Path,
    *,
    training_profile: VLAProfile = "full",
    feature_layer: int = DEFAULT_FEATURE_LAYER,
) -> Path:
    """Atomically save only the trainable head and its frozen-VLM reference."""
    if not model_id.strip() or not model_revision.strip():
        raise ValueError("VLA checkpoint model ID and revision must be non-empty")
    if training_profile not in {"full", "limited"}:
        raise ValueError("VLA training_profile must be 'full' or 'limited'")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_id": model_id,
            "model_revision": model_revision,
            "training_profile": training_profile,
            "feature_layer": feature_layer,
            "chunk_size": head.chunk_size,
            "hidden_size": head.hidden_size,
            "image_tokens": head.image_tokens,
            "state_dict": _copy_state_dict(head),
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_vla_training_head(
    path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    training_profile: VLAProfile,
    hidden_size: int,
    feature_layer: int,
    chunk_size: int,
) -> VLAActionHead:
    """Load a head checkpoint only when its frozen-VLM metadata matches."""
    checkpoint = _read_checkpoint(path)
    if checkpoint.model_id != model_id:
        raise ValueError("Warm-start VLA checkpoint uses a different model")
    if checkpoint.model_revision != model_revision:
        raise ValueError("Warm-start VLA checkpoint uses a different model revision")
    if checkpoint.training_profile != training_profile:
        raise ValueError("Warm-start VLA checkpoint uses a different training profile")
    if checkpoint.hidden_size != hidden_size:
        raise ValueError("Warm-start VLA checkpoint has a different feature size")
    if checkpoint.feature_layer != feature_layer:
        raise ValueError("Warm-start VLA checkpoint uses a different feature layer")
    if checkpoint.chunk_size != chunk_size:
        raise ValueError("Warm-start VLA checkpoint predicts a different chunk size")
    head = VLAActionHead(
        checkpoint.hidden_size,
        checkpoint.image_tokens,
        chunk_size=checkpoint.chunk_size,
    )
    head.load_state_dict(checkpoint.state_dict, strict=True)
    return head


def load_vla_policy(
    path: str | Path,
    device: str | torch.device = "auto",
    *,
    replan_interval: int | None = None,
    ensemble_decay: float = 0.0,
) -> VLAActionPolicy:
    """Restore a head checkpoint as a lazy-loading shared evaluation policy.

    ``replan_interval`` defaults to 2 steps, or 1 for single-step heads.
    """
    checkpoint = _read_checkpoint(path)
    head = VLAActionHead(
        checkpoint.hidden_size,
        checkpoint.image_tokens,
        chunk_size=checkpoint.chunk_size,
    )
    head.load_state_dict(checkpoint.state_dict, strict=True)
    return VLAActionPolicy(
        None,
        None,
        model_id=checkpoint.model_id,
        model_revision=checkpoint.model_revision,
        head_state=checkpoint.state_dict,
        image_tokens=checkpoint.image_tokens,
        training_profile=checkpoint.training_profile,
        feature_layer=checkpoint.feature_layer,
        chunk_size=checkpoint.chunk_size,
        replan_interval=replan_interval,
        ensemble_decay=ensemble_decay,
        device=device,
    )


class _CheckpointContents(NamedTuple):
    """Hold validated metadata and weights from a VLA head checkpoint."""

    model_id: str
    model_revision: str
    training_profile: VLAProfile
    feature_layer: int
    chunk_size: int
    hidden_size: int
    image_tokens: int
    state_dict: Mapping[str, Tensor]


def _read_checkpoint(path: str | Path) -> _CheckpointContents:
    """Read a token-grid VLA checkpoint and reject superseded heads.

    Format 3 heads predict a single step and load with a chunk size of 1.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("VLA checkpoint must contain a mapping")
    format_version = checkpoint.get("format_version")
    if isinstance(format_version, bool) or format_version not in {
        _SINGLE_STEP_CHECKPOINT_FORMAT,
        CHECKPOINT_FORMAT_VERSION,
    }:
        raise ValueError(
            f"VLA checkpoint format {format_version!r} is not supported; formats "
            "before 3 used the superseded last-token head, so retrain the head"
        )
    model_id = checkpoint.get("model_id")
    model_revision = checkpoint.get("model_revision")
    training_profile = checkpoint.get("training_profile")
    integers = {
        name: checkpoint.get(name)
        for name in ("feature_layer", "hidden_size", "image_tokens")
    }
    integers["chunk_size"] = (
        1
        if format_version == _SINGLE_STEP_CHECKPOINT_FORMAT
        else checkpoint.get("chunk_size")
    )
    state_dict = checkpoint.get("state_dict")
    if (
        not isinstance(model_id, str)
        or not isinstance(model_revision, str)
        or training_profile not in {"full", "limited"}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integers.values()
        )
        or integers["feature_layer"] < 0
        or integers["hidden_size"] < 1
        or integers["image_tokens"] < 1
        or integers["chunk_size"] < 1
        or not isinstance(state_dict, Mapping)
    ):
        raise ValueError("VLA checkpoint metadata is invalid")
    if format_version == _SINGLE_STEP_CHECKPOINT_FORMAT:
        state_dict = {
            name: (
                value.reshape(1, 4) if name in {"delta_mean", "delta_std"} else value
            )
            for name, value in state_dict.items()
        }
    return _CheckpointContents(
        model_id=model_id,
        model_revision=model_revision,
        training_profile=training_profile,
        feature_layer=integers["feature_layer"],
        chunk_size=integers["chunk_size"],
        hidden_size=integers["hidden_size"],
        image_tokens=integers["image_tokens"],
        state_dict=state_dict,
    )


def _load_or_encode(
    dataset: Dataset[dict[str, Any]],
    encoder: VLAEncoder,
    config: VLAConfig,
    split: Literal["train", "validation"],
    feature_cache: VLAFeatureCache | None,
    *,
    expected_actions: Tensor,
) -> tuple[EncodedVLADataset, float, bool]:
    """Reuse matching cached features or encode one split and cache it.

    A cached split is used only if it reproduces ``expected_actions``, the
    dataset's current clean action labels, row for row.
    """
    cache_path: Path | None = None
    key: dict[str, Any] = {}
    if feature_cache is not None:
        if split not in feature_cache.split_keys:
            raise ValueError(f"VLA feature cache has no key for the {split} split")
        key = {
            "cache_version": _FEATURE_CACHE_VERSION,
            "split": split,
            "data": feature_cache.split_keys[split],
            "model_id": encoder.model_id,
            "model_revision": encoder.model_revision,
            "transformers_version": encoder.transformers_version,
            "model_dtype": encoder.model_dtype,
            "feature_layer": encoder.feature_layer,
            "prompt_prefix": _PROMPT_PREFIX,
            "prompt_layout": _PROMPT_LAYOUT,
            "image_splitting": False,
            "max_encoding_samples": config.max_encoding_samples,
        }
        serialized_key = json.dumps(key, sort_keys=True)
        digest = hashlib.sha256(serialized_key.encode("utf-8")).hexdigest()[:16]
        cache_path = feature_cache.directory / f"{split}-{digest}.pt"
        key = json.loads(serialized_key)
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            cached_actions = (
                cached.get("actions") if isinstance(cached, Mapping) else None
            )
            if (
                isinstance(cached, Mapping)
                and cached.get("key") == key
                and isinstance(cached_actions, Tensor)
                and torch.equal(cached_actions, expected_actions)
            ):
                encoded = EncodedVLADataset(
                    cached["grid"],
                    cached["context"],
                    cached["states"],
                    cached["actions"],
                    cached["episode_indices"],
                )
                return encoded, float(cached["encoding_seconds"]), True
    started = time.perf_counter()
    encoded = _encode_dataset(
        dataset,
        encoder,
        batch_size=config.encoding_batch_size,
        num_workers=config.num_workers,
        max_samples=config.max_encoding_samples,
    )
    _synchronize(encoder.device)
    seconds = time.perf_counter() - started
    if not torch.equal(encoded.actions, expected_actions):
        raise ValueError("Encoded VLA rows do not match the dataset's action labels")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(
            {
                "key": key,
                "grid": encoded.grid,
                "context": encoded.context,
                "states": encoded.states,
                "actions": encoded.actions,
                "episode_indices": encoded.episode_indices,
                "encoding_seconds": seconds,
            },
            temporary,
        )
        os.replace(temporary, cache_path)
    return encoded, seconds, False


def _encode_dataset(
    dataset: Dataset[dict[str, Any]],
    encoder: VLAEncoder,
    *,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> EncodedVLADataset:
    """Encode one split in bounded batches into preallocated CPU tensors."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_vla_samples,
        persistent_workers=num_workers > 0,
    )
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    grid_store: Tensor | None = None
    context_store = torch.empty((total, encoder.hidden_size), dtype=torch.float32)
    states = torch.empty((total, 12), dtype=torch.float32)
    actions = torch.empty((total, 5), dtype=torch.float32)
    episodes = torch.empty(total, dtype=torch.int64)
    encoded_count = 0
    for batch in tqdm(loader, desc="Encoding frozen VLM", unit="batch"):
        remaining = total - encoded_count
        if remaining <= 0:
            break
        batch = {name: value[:remaining] for name, value in batch.items()}
        features = encoder.encode(batch["front"], batch["wrist"], batch["instruction"])
        count = len(batch["action"])
        if (
            features.grid.ndim != 4
            or features.grid.shape[:2] != (count, _CAMERA_COUNT)
            or features.grid.shape[3] != encoder.hidden_size
            or features.context.shape != (count, encoder.hidden_size)
        ):
            raise ValueError("VLA encoder returned an invalid feature batch")
        if (
            not torch.isfinite(features.grid).all()
            or not torch.isfinite(features.context).all()
        ):
            raise FloatingPointError("VLA encoder returned non-finite features")
        if grid_store is None:
            # fp16 halves host memory; the frozen VLM already runs in fp16 on MPS.
            grid_store = torch.empty(
                (total, *features.grid.shape[1:]),
                dtype=torch.float16,
            )
        elif features.grid.shape[1:] != grid_store.shape[1:]:
            raise ValueError("VLA encoder returned inconsistent image-token grids")
        stop = encoded_count + count
        grid_store[encoded_count:stop] = features.grid.detach().to("cpu")
        context_store[encoded_count:stop] = features.context.detach().to("cpu")
        states[encoded_count:stop] = batch["state"].to(dtype=torch.float32)
        actions[encoded_count:stop] = batch["action"].to(dtype=torch.float32)
        episodes[encoded_count:stop] = batch["episode_index"].to(dtype=torch.int64)
        encoded_count = stop
    if grid_store is None or encoded_count != total:
        raise ValueError("VLA feature encoding produced no complete split")
    if not torch.isfinite(grid_store).all():
        raise FloatingPointError("VLA image-token features overflow float16")
    return EncodedVLADataset(grid_store, context_store, states, actions, episodes)


def _collate_vla_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch images as lists and stack only the numeric training fields."""
    return {
        "front": [sample["front"] for sample in samples],
        "wrist": [sample["wrist"] for sample in samples],
        "instruction": [sample["instruction"] for sample in samples],
        "state": torch.stack([sample["state"] for sample in samples]),
        "action": torch.stack([sample["action"] for sample in samples]),
        "episode_index": torch.stack([sample["episode_index"] for sample in samples]),
    }


def _chunk_labels(
    dataset: Dataset[dict[str, Any]],
    config: VLAConfig,
) -> tuple[Tensor, Tensor]:
    """Read one split's action chunks, truncated like its encoded features."""
    chunk_method = getattr(dataset, "action_chunks", None)
    if not callable(chunk_method):
        raise TypeError("VLA training datasets must provide action_chunks()")
    chunks, mask = chunk_method(config.chunk_size)
    count = len(dataset)
    if (
        not isinstance(chunks, Tensor)
        or not isinstance(mask, Tensor)
        or chunks.shape != (count, config.chunk_size, 5)
        or mask.shape != (count, config.chunk_size)
        or mask.dtype != torch.bool
    ):
        raise ValueError("VLA action chunks have an invalid shape")
    rows = (
        count
        if config.max_encoding_samples is None
        else min(count, config.max_encoding_samples)
    )
    return chunks[:rows].to(torch.float32), mask[:rows]


def _training_statistics(encoded: EncodedVLADataset) -> VLANormalization:
    """Compute input and residual-target statistics from encoded training rows."""
    grid_sum = torch.zeros(
        encoded.grid.shape[1], encoded.grid.shape[3], dtype=torch.float64
    )
    grid_square_sum = torch.zeros_like(grid_sum)
    token_count = 0
    for start in range(0, len(encoded), 128):
        chunk = encoded.grid[start : start + 128].to(torch.float64)
        grid_sum += chunk.sum(dim=(0, 2))
        grid_square_sum += chunk.square().sum(dim=(0, 2))
        token_count += chunk.shape[0] * chunk.shape[2]
    grid_mean = grid_sum / token_count
    grid_std = (
        (grid_square_sum / token_count - grid_mean.square()).clamp_min(0.0).sqrt()
    )
    # Each chunk position has its own statistics: later steps move farther.
    deltas = (encoded.action_chunks[..., :4] - encoded.states[:, None, _TCP_POSE]).to(
        torch.float64
    )
    weights = encoded.chunk_mask.to(torch.float64).unsqueeze(-1)
    counts = weights.sum(dim=0).clamp_min(1.0)
    delta_mean = (deltas * weights).sum(dim=0) / counts
    delta_std = (((deltas - delta_mean) ** 2 * weights).sum(dim=0) / counts).sqrt()
    return VLANormalization(
        state_mean=encoded.states.mean(dim=0),
        state_std=encoded.states.std(dim=0, unbiased=False).clamp_min(_STATE_STD_FLOOR),
        grid_mean=grid_mean.to(torch.float32),
        grid_std=grid_std.to(torch.float32).clamp_min(_FEATURE_STD_FLOOR),
        context_mean=encoded.context.mean(dim=0),
        context_std=encoded.context.std(dim=0, unbiased=False).clamp_min(
            _FEATURE_STD_FLOOR
        ),
        delta_mean=delta_mean.to(torch.float32),
        delta_std=delta_std.to(torch.float32).clamp_min(_STATE_STD_FLOOR),
    )


def _batch_to_device(
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Move one encoded batch and its action-chunk labels to the head device."""
    return (
        batch["grid"].to(device=device, dtype=torch.float32),
        batch["context"].to(device=device, dtype=torch.float32),
        batch["state"].to(device=device, dtype=torch.float32),
        batch["action_chunk"].to(device=device, dtype=torch.float32),
        batch["chunk_mask"].to(device=device),
    )


def _chunk_losses(
    head: VLAActionHead,
    output: Tensor,
    state: Tensor,
    action_chunk: Tensor,
    chunk_mask: Tensor,
    gripper_positive_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return masked L1 on normalized deltas and masked weighted gripper BCE."""
    deltas, logits = head.split_output(output)
    weights = chunk_mask.to(output.dtype)
    valid = weights.sum().clamp_min(1.0)
    delta_error = (deltas - head.delta_targets(state, action_chunk)).abs().mean(dim=-1)
    gripper_target = _binary_gripper_targets(action_chunk[..., 4].reshape(-1))
    gripper_bce = F.binary_cross_entropy_with_logits(
        logits,
        gripper_target.reshape_as(logits),
        pos_weight=gripper_positive_weight,
        reduction="none",
    )
    return (delta_error * weights).sum() / valid, (gripper_bce * weights).sum() / valid


def _action_errors(prediction: Tensor, action: Tensor) -> Tensor:
    """Return absolute xyz/yaw errors and binary gripper-decision errors."""
    gripper_target = _binary_gripper_targets(action[:, 4])
    predicted = torch.cat(
        (
            prediction[:, :4],
            (prediction[:, 4] >= 0.0).to(action.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    target = torch.cat((action[:, :4], gripper_target.unsqueeze(1)), dim=1)
    return (predicted - target).abs()


def _evaluate_head(
    head: VLAActionHead,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    gripper_loss_weight: float,
    gripper_positive_weight: Tensor,
    ablate_features: bool = False,
) -> _ValidationMetrics:
    """Measure action error, gripper classification, and training objective.

    Action errors and gripper accuracy describe the step executed immediately;
    ``chunk_l1`` averages the error over every valid position of the chunk.
    With ``ablate_features``, every frozen-VLM input is replaced by its
    training mean, so the gap to the normal result shows how much the head
    relies on vision and language rather than robot state alone.
    """
    was_training = head.training
    head.eval()
    absolute_errors: list[Tensor] = []
    episode_errors: dict[int, list[float]] = {}
    objective_loss_sum = 0.0
    gripper_bce_sum = 0.0
    chunk_error_sum = 0.0
    chunk_position_count = 0
    gripper_correct = 0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            grid, context, state, action_chunk, chunk_mask = _batch_to_device(
                batch, device
            )
            if ablate_features:
                grid = head.grid_mean[None, :, None, :].expand_as(grid)
                context = head.context_mean.expand_as(context)
            output = head(grid, context, state)
            predicted_chunk = head.absolute_actions(output, state)
            action = action_chunk[:, 0]
            error = _action_errors(predicted_chunk[:, 0], action).cpu()
            if not torch.isfinite(error).all():
                raise FloatingPointError("VLA validation produced non-finite errors")
            chunk_errors = _action_errors(
                predicted_chunk.reshape(-1, 5),
                action_chunk.reshape(-1, 5),
            ).mean(dim=1)
            valid_positions = chunk_mask.reshape(-1)
            chunk_error_sum += float(chunk_errors[valid_positions].sum())
            chunk_position_count += int(valid_positions.sum())
            action_loss, gripper_bce = _chunk_losses(
                head,
                output,
                state,
                action_chunk,
                chunk_mask,
                gripper_positive_weight,
            )
            objective_loss_sum += float(
                (1.0 - gripper_loss_weight) * action_loss
                + gripper_loss_weight * gripper_bce
            ) * len(action)
            gripper_bce_sum += float(gripper_bce) * len(action)
            gripper_target = _binary_gripper_targets(action[:, 4])
            gripper_correct += int(
                ((predicted_chunk[:, 0, 4] >= 0.0) == gripper_target.bool())
                .sum()
                .item()
            )
            sample_count += len(action)
            absolute_errors.append(error)
            for episode_index, sample_error in zip(
                batch["episode_index"].tolist(), error.mean(dim=1).tolist(), strict=True
            ):
                episode_errors.setdefault(int(episode_index), []).append(sample_error)
    head.train(was_training)
    if not absolute_errors:
        raise ValueError("VLA validation dataset must be non-empty")
    errors = torch.cat(absolute_errors)
    return _ValidationMetrics(
        l1=float(errors.mean()),
        chunk_l1=chunk_error_sum / max(chunk_position_count, 1),
        action_mae={
            name: float(value)
            for name, value in zip(_ACTION_NAMES, errors.mean(dim=0), strict=True)
        },
        episode_l1={
            str(index): float(np.mean(values))
            for index, values in sorted(episode_errors.items())
        },
        objective_loss=objective_loss_sum / sample_count,
        gripper_bce=gripper_bce_sum / sample_count,
        gripper_accuracy=gripper_correct / sample_count,
    )


def _binary_gripper_targets(gripper_values: Tensor) -> Tensor:
    """Convert open-fraction labels to closed/open classes at the 0.5 boundary."""
    if gripper_values.ndim != 1 or not torch.isfinite(gripper_values).all():
        raise ValueError("Gripper labels must be a finite vector")
    return (gripper_values >= 0.5).to(dtype=gripper_values.dtype)


class _ValidationMetrics(BaseModel):
    """Hold validation losses during action-head training."""

    l1: float
    chunk_l1: float
    action_mae: dict[str, float]
    episode_l1: dict[str, float]
    objective_loss: float
    gripper_bce: float
    gripper_accuracy: float


def _copy_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Copy model tensors to CPU for safe, compact checkpointing."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Validate and copy one numeric vector to float32."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != shape or not np.isfinite(vector).all():
        raise ValueError(f"VLA {name} must be a finite vector with shape {shape}")
    return vector


def _rgb_image(value: Any, name: str) -> np.ndarray:
    """Convert one RGB camera frame to a validated uint8 HWC array."""
    if value is None:
        raise ValueError(f"VLA observations require {name}")
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be a 3-D RGB image")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape != (256, 256, 3):
        raise ValueError(f"{name} must have shape (256, 256, 3)")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f"{name} must use uint8 RGB pixels")
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(f"{name} float pixels must be finite in [0, 1]")
        image = np.rint(image * 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def _move_tensor(value: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Move processor tensors to the model device and cast only floating fields."""
    if value.is_floating_point():
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def _model_dtype(device: torch.device) -> torch.dtype:
    """Select a compact inference dtype on accelerators and fp32 on CPU."""
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _synchronize(device: torch.device) -> None:
    """Wait for asynchronous device work before measuring encoder throughput."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_memory_mb(device: torch.device) -> float | None:
    """Return accelerator allocation measured by the available backend API."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return float(torch.mps.driver_allocated_memory() / (1024**2))
    return None


def _reset_peak_memory(device: torch.device) -> None:
    """Reset accelerator peak tracking immediately before encoder measurement."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
