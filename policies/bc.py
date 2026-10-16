"""Behavior-cloning models, LeRobot adapters, training, and policy wrappers."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from eval.policy import Policy
from motion.control import WORKSPACE_LOWER, WORKSPACE_UPPER

BCPolicyType = Literal["state", "image"]
OBJECT_POSE_NAMES = ("cube_red", "cube_blue", "cube_green", "bowl")
NUMERIC_FEATURES = (
    "observation.state",
    "observation.object_poses",
    "observation.target_pose",
    "action",
)
IMAGE_FEATURES = ("observation.images.front", "observation.images.wrist")
ACTION_COMPONENT_NAMES = ("x", "y", "z", "yaw", "gripper_open")
_ACTION_LOW = np.array([*WORKSPACE_LOWER, -np.pi / 2, 0.0], dtype=np.float32)
_ACTION_HIGH = np.array([*WORKSPACE_UPPER, np.pi / 2, 1.0], dtype=np.float32)
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)
_LOGGER = logging.getLogger(__name__)


class BCTrainingConfig(BaseModel):
    """Validate hyperparameters shared by the state and image BC trainers."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0.0, allow_inf_nan=False)
    num_workers: int = Field(default=0, ge=0)
    device: str = "auto"
    seed: int = Field(default=42, ge=0)
    patience: int = Field(default=5, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    pretrained: bool = True


class BCEpochMetrics(BaseModel):
    """Record one epoch's mean L1 training and validation losses."""

    epoch: int
    training_l1: float
    validation_l1: float


class BCTrainingResult(BaseModel):
    """Summarize validation progress and the best selected model epoch."""

    device: str
    initial_validation_l1: float
    best_validation_l1: float
    best_epoch: int
    history: list[BCEpochMetrics]
    validation_action_mae: dict[str, float]
    validation_episode_l1: dict[str, float]


class BCValidationMetrics(BaseModel):
    """Report validation action error by component and source episode."""

    l1: float
    action_mae: dict[str, float]
    episode_l1: dict[str, float]


class LeRobotBCDataset(Dataset[dict[str, Tensor]]):
    """Expose LeRobot frames as compact, consistently typed BC samples."""

    def __init__(self, dataset: Any, policy_type: BCPolicyType) -> None:
        """Wrap one read-only LeRobot dataset for a state or image policy."""
        self.policy_type = policy_type
        self._dataset = dataset
        self._numeric = dataset.select_columns(
            [*NUMERIC_FEATURES, "episode_index"]
        ).with_format("numpy")
        self._statistics = self._numeric[:]

    def __len__(self) -> int:
        """Return the number of transitions in the wrapped LeRobot dataset."""
        return len(self._numeric)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one BC transition, decoding camera frames only for image BC."""
        sample = self._numeric_sample(index)
        if self.policy_type == "image":
            frame = self._dataset[index]
            sample["front"] = _image_tensor(frame[IMAGE_FEATURES[0]])
            sample["wrist"] = _image_tensor(frame[IMAGE_FEATURES[1]])
        return sample

    def _numeric_sample(self, index: int) -> dict[str, Tensor]:
        """Return the numeric columns without decoding any camera video."""
        row = self._numeric[index]
        return {
            "state": _as_float_tensor(row["observation.state"]),
            "object_poses": _as_float_tensor(row["observation.object_poses"]),
            "target_pose": _as_float_tensor(row["observation.target_pose"]),
            "action": _as_float_tensor(row["action"]),
            "episode_index": torch.tensor(row["episode_index"], dtype=torch.int64),
        }

    def normalization_features(self) -> Tensor:
        """Return training-only numeric inputs without decoding video frames."""
        state = np.asarray(self._statistics["observation.state"], dtype=np.float32)
        if self.policy_type == "image":
            values = state
        else:
            values = np.concatenate(
                (
                    state,
                    np.asarray(
                        self._statistics["observation.object_poses"], dtype=np.float32
                    ),
                    np.asarray(
                        self._statistics["observation.target_pose"], dtype=np.float32
                    ),
                ),
                axis=1,
            )
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True))


class StateBC(nn.Module):
    """MLP policy over proprioception, oracle object poses, and the target pose."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (256, 256)) -> None:
        """Build a normalized 44-D state regressor for 5-D actions."""
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(44, dtype=torch.float32))
        self.register_buffer("input_std", torch.ones(44, dtype=torch.float32))
        layers: list[nn.Module] = []
        input_size = 44
        for hidden_size in hidden_sizes:
            layers.extend(
                (
                    nn.Linear(input_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                )
            )
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 5))
        self.network = nn.Sequential(*layers)

    def set_input_statistics(self, mean: Tensor, std: Tensor) -> None:
        """Set train-split normalization statistics for the concatenated input."""
        if mean.shape != self.input_mean.shape or std.shape != self.input_std.shape:
            raise ValueError("State BC normalization must contain 44 values")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(1e-6))

    def forward(
        self,
        state: Tensor,
        object_poses: Tensor,
        target_pose: Tensor,
    ) -> Tensor:
        """Predict the next absolute 5-D action from oracle state inputs."""
        features = torch.cat((state, object_poses, target_pose), dim=-1)
        return self.network((features - self.input_mean) / self.input_std)


class ImageBC(nn.Module):
    """Two-view frozen ImageNet ResNet-18 features fused with proprioception."""

    def __init__(self, *, pretrained: bool = True) -> None:
        """Build frozen pretrained camera encoders and a trainable action head."""
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.front_encoder = _resnet_encoder(weights)
        self.wrist_encoder = _resnet_encoder(weights)
        self.register_buffer("state_mean", torch.zeros(12, dtype=torch.float32))
        self.register_buffer("state_std", torch.ones(12, dtype=torch.float32))
        self.register_buffer(
            "image_mean",
            torch.tensor(_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std", torch.tensor(_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(1024 + 12, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 5),
        )
        self.front_encoder.requires_grad_(False)
        self.wrist_encoder.requires_grad_(False)
        self.front_encoder.eval()
        self.wrist_encoder.eval()

    def train(self, mode: bool = True) -> ImageBC:
        """Train the action head while keeping frozen ResNet encoders in eval mode."""
        super().train(mode)
        self.front_encoder.eval()
        self.wrist_encoder.eval()
        return self

    def set_input_statistics(self, mean: Tensor, std: Tensor) -> None:
        """Set train-split normalization statistics for proprioception."""
        if mean.shape != self.state_mean.shape or std.shape != self.state_std.shape:
            raise ValueError("Image BC state normalization must contain 12 values")
        self.state_mean.copy_(mean)
        self.state_std.copy_(std.clamp_min(1e-6))

    def forward(self, front: Tensor, wrist: Tensor, state: Tensor) -> Tensor:
        """Predict the next absolute action from both camera views and state."""
        visual_features = self.encode_visual(front, wrist)
        return self.forward_visual_features(visual_features, state)

    def encode_visual(self, front: Tensor, wrist: Tensor) -> Tensor:
        """Extract frozen front and wrist ResNet features from normalized RGB inputs."""
        front_image = _normalize_image_batch(front, self.image_mean, self.image_std)
        wrist_image = _normalize_image_batch(wrist, self.image_mean, self.image_std)
        with torch.no_grad():
            return torch.cat(
                (self.front_encoder(front_image), self.wrist_encoder(wrist_image)),
                dim=-1,
            )

    def forward_visual_features(self, visual_features: Tensor, state: Tensor) -> Tensor:
        """Predict an action from cached image embeddings and proprioception."""
        normalized_state = (state - self.state_mean) / self.state_std
        return self.head(torch.cat((visual_features, normalized_state), dim=-1))


class CachedImageBCDataset(Dataset[dict[str, Tensor]]):
    """Store one frozen visual embedding per image-BC transition."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Tensor]],
        visual_features: Tensor,
    ) -> None:
        """Pair cached embeddings with numeric values from the source split."""
        if visual_features.shape != (len(dataset), 1024):
            raise ValueError("Cached ResNet features must have shape [N, 1024]")
        self._dataset = dataset
        self._visual_features = visual_features.to(device="cpu", dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of cached image transitions."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return cached visual features and numeric policy inputs without decoding."""
        numeric_sample = getattr(self._dataset, "_numeric_sample", None)
        if callable(numeric_sample):
            sample = numeric_sample(index)
        else:
            sample = {
                key: value
                for key, value in self._dataset[index].items()
                if key
                in {"state", "object_poses", "target_pose", "action", "episode_index"}
            }
        sample["visual_features"] = self._visual_features[index]
        return sample

    def normalization_features(self) -> Tensor:
        """Reuse the source split's numeric features for state normalization."""
        feature_method = getattr(self._dataset, "normalization_features", None)
        if not callable(feature_method):
            raise TypeError("Source dataset must provide normalization_features()")
        return feature_method()


class StateBCPolicy:
    """Evaluation adapter for a trained state-based behavior-cloning model."""

    def __init__(self, model: StateBC, device: str | torch.device = "auto") -> None:
        """Move the model to the selected inference device and enter eval mode."""
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()

    def reset(self, instruction: str | None = None) -> None:
        """Satisfy the shared policy protocol; BC has no temporal state."""

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Predict a clipped action from the environment's oracle observation."""
        state = _observation_vector(obs, "observation.state", (12,))
        object_poses = _flatten_object_poses(obs.get("object_poses"))
        target_pose = _observation_vector(obs, "target_pose", (4,))
        with torch.inference_mode():
            action = self.model(
                torch.from_numpy(state).unsqueeze(0).to(self.device),
                torch.from_numpy(object_poses).unsqueeze(0).to(self.device),
                torch.from_numpy(target_pose).unsqueeze(0).to(self.device),
            )
        return _clip_action(action[0].cpu().numpy())


class ImageBCPolicy:
    """Evaluation adapter for a trained front/wrist image BC model."""

    def __init__(self, model: ImageBC, device: str | torch.device = "auto") -> None:
        """Move the model to the selected inference device and enter eval mode."""
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()

    def reset(self, instruction: str | None = None) -> None:
        """Satisfy the shared policy protocol; BC has no temporal state."""

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Predict a clipped action from the camera images and proprioception."""
        state = _observation_vector(obs, "observation.state", (12,))
        front = _image_tensor(obs.get(IMAGE_FEATURES[0]))
        wrist = _image_tensor(obs.get(IMAGE_FEATURES[1]))
        with torch.inference_mode():
            action = self.model(
                front.unsqueeze(0).to(self.device),
                wrist.unsqueeze(0).to(self.device),
                torch.from_numpy(state).unsqueeze(0).to(self.device),
            )
        return _clip_action(action[0].cpu().numpy())


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Prefer Apple MPS, then CUDA, and otherwise use CPU for BC workloads."""
    if isinstance(device, torch.device):
        selected = device
    elif device == "auto":
        if torch.backends.mps.is_available():
            selected = torch.device("mps")
        elif torch.cuda.is_available():
            selected = torch.device("cuda")
        else:
            selected = torch.device("cpu")
    else:
        selected = torch.device(device)
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def train_behavior_cloning(
    policy_type: BCPolicyType,
    train_dataset: Dataset[dict[str, Tensor]],
    validation_dataset: Dataset[dict[str, Tensor]],
    config: BCTrainingConfig,
) -> tuple[StateBC | ImageBC, BCTrainingResult]:
    """Train a BC model with L1 loss and restore the best validation checkpoint."""
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must both be non-empty")
    torch.manual_seed(config.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.seed)
    device = resolve_device(config.device)
    model: StateBC | ImageBC = (
        StateBC() if policy_type == "state" else ImageBC(pretrained=config.pretrained)
    )
    model.to(device)
    if policy_type == "image" and isinstance(model, ImageBC):
        train_dataset = _cache_visual_features(train_dataset, model, config, device)
        validation_dataset = _cache_visual_features(
            validation_dataset, model, config, device
        )
    _set_model_normalization(model, policy_type, train_dataset)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = _data_loader(
        train_dataset, config, shuffle=True, generator=generator
    )
    validation_loader = _data_loader(
        validation_dataset, config, shuffle=False, generator=None
    )
    initial_validation_l1 = _evaluate(model, policy_type, validation_loader, device).l1
    best_validation_l1 = initial_validation_l1
    best_epoch = 0
    best_state = _copy_model_state(model)
    epochs_without_improvement = 0
    history: list[BCEpochMetrics] = []

    epoch_progress = tqdm(
        range(1, config.epochs + 1),
        desc=f"{policy_type} BC",
        unit="epoch",
    )
    for epoch in epoch_progress:
        model.train()
        training_l1 = _train_epoch(
            model, policy_type, train_loader, optimizer, device, config.max_grad_norm
        )
        validation_l1 = _evaluate(model, policy_type, validation_loader, device).l1
        history.append(
            BCEpochMetrics(
                epoch=epoch,
                training_l1=training_l1,
                validation_l1=validation_l1,
            )
        )
        epoch_progress.set_postfix(
            train_l1=f"{training_l1:.4f}",
            val_l1=f"{validation_l1:.4f}",
        )
        if validation_l1 < best_validation_l1:
            best_validation_l1 = validation_l1
            best_epoch = epoch
            best_state = _copy_model_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _evaluate(model, policy_type, validation_loader, device)
    return model, BCTrainingResult(
        device=str(device),
        initial_validation_l1=initial_validation_l1,
        best_validation_l1=best_validation_l1,
        best_epoch=best_epoch,
        history=history,
        validation_action_mae=validation_metrics.action_mae,
        validation_episode_l1=validation_metrics.episode_l1,
    )


def save_bc_checkpoint(
    model: StateBC | ImageBC,
    policy_type: BCPolicyType,
    path: str | Path,
) -> Path:
    """Atomically save model weights in a versioned, weights-only checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "policy_type": policy_type,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_bc_policy(
    path: str | Path,
    device: str | torch.device = "auto",
) -> Policy:
    """Load a state or image BC checkpoint into the shared evaluation protocol."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported BC checkpoint format")
    policy_type = checkpoint.get("policy_type")
    if policy_type == "state":
        model: StateBC | ImageBC = StateBC()
        policy: Policy = StateBCPolicy(model, device)
    elif policy_type == "image":
        model = ImageBC(pretrained=False)
        policy = ImageBCPolicy(model, device)
    else:
        raise ValueError(f"Unsupported BC policy type: {policy_type!r}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return policy


def _resnet_encoder(weights: ResNet18_Weights | None) -> nn.Module:
    """Build one ResNet-18 feature encoder without its classification layer."""
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    return encoder


def _as_float_tensor(value: Any) -> Tensor:
    """Copy one numeric sample into a finite float32 tensor."""
    tensor = torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("BC features must contain only finite values")
    return tensor


def _image_tensor(value: Any) -> Tensor:
    """Convert RGB data to float32 CHW in the [0, 1] range."""
    if value is None:
        raise ValueError("Image BC observations require both camera images")
    image = torch.as_tensor(np.asarray(value).copy())
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB image, got shape {tuple(image.shape)}")
    if image.shape[0] == 3:
        pass
    elif image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected three RGB channels, got shape {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.to(torch.float32).div_(255.0)
    else:
        image = image.to(torch.float32)
    if image.shape[1:] != (128, 128):
        raise ValueError(f"Expected a 128x128 image, got shape {tuple(image.shape)}")
    return image


def _normalize_image_batch(image: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Normalize a batch of CHW or HWC images with ImageNet statistics."""
    if image.ndim != 4:
        raise ValueError(f"Expected a batched image tensor, got {tuple(image.shape)}")
    if image.shape[1] == 3:
        pass
    elif image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected three image channels, got {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.to(torch.float32).div_(255.0)
    else:
        image = image.to(torch.float32)
    return (image - mean) / std


def _observation_vector(
    observation: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Validate and convert a vector field from a Gymnasium observation."""
    value = np.asarray(observation.get(key), dtype=np.float32)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must be a finite vector with shape {shape}")
    return value


def _flatten_object_poses(value: Any) -> np.ndarray:
    """Flatten named cube and bowl poses in the same order as Phase 2.1 data."""
    if not isinstance(value, Mapping):
        raise TypeError("State BC requires an object_poses mapping")
    components: list[np.ndarray] = []
    for name in OBJECT_POSE_NAMES:
        pose = value.get(name)
        if not isinstance(pose, Mapping):
            raise KeyError(f"object_poses is missing {name!r}")
        position = np.asarray(pose.get("position"), dtype=np.float32)
        quaternion = np.asarray(pose.get("quaternion_wxyz"), dtype=np.float32)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError(
                f"object_poses[{name!r}] has invalid position or quaternion"
            )
        components.extend((position, quaternion))
    flattened = np.concatenate(components)
    if not np.all(np.isfinite(flattened)):
        raise ValueError("object_poses must contain only finite values")
    return flattened


def _clip_action(action: np.ndarray) -> np.ndarray:
    """Enforce the common workspace, yaw, and gripper command limits."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (5,) or not np.all(np.isfinite(value)):
        raise ValueError("BC model must produce a finite 5-D action")
    return np.clip(value, _ACTION_LOW, _ACTION_HIGH)


def _set_model_normalization(
    model: StateBC | ImageBC,
    policy_type: BCPolicyType,
    dataset: Dataset[dict[str, Tensor]],
) -> None:
    """Fit input statistics from the training split only."""
    feature_method = getattr(dataset, "normalization_features", None)
    if not callable(feature_method):
        raise TypeError("Training datasets must provide normalization_features()")
    features = feature_method().to(dtype=torch.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Training normalization features must be a non-empty matrix")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    model_matches_policy = (
        policy_type == "state"
        and isinstance(model, StateBC)
        or policy_type == "image"
        and isinstance(model, ImageBC)
    )
    if not model_matches_policy:
        raise TypeError("Policy type and model class do not match")
    model.set_input_statistics(mean, std)


def _data_loader(
    dataset: Dataset[dict[str, Tensor]],
    config: BCTrainingConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader[dict[str, Tensor]]:
    """Build a bounded-memory loader suitable for CPU, MPS, or CUDA."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def _predict_batch(
    model: StateBC | ImageBC,
    policy_type: BCPolicyType,
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Move one batch to the device and return prediction and action label."""
    action = batch["action"].to(device, non_blocking=device.type == "cuda")
    state = batch["state"].to(device, non_blocking=device.type == "cuda")
    if policy_type == "state" and isinstance(model, StateBC):
        prediction = model(
            state,
            batch["object_poses"].to(device),
            batch["target_pose"].to(device),
        )
    elif policy_type == "image" and isinstance(model, ImageBC):
        if "visual_features" in batch:
            prediction = model.forward_visual_features(
                batch["visual_features"].to(device),
                state,
            )
        else:
            prediction = model(
                batch["front"].to(device),
                batch["wrist"].to(device),
                state,
            )
    else:
        raise TypeError("Policy type and model class do not match")
    return prediction, action


def _cache_visual_features(
    dataset: Dataset[dict[str, Tensor]],
    model: ImageBC,
    config: BCTrainingConfig,
    device: torch.device,
) -> CachedImageBCDataset:
    """Decode each image once and cache frozen ResNet embeddings for training."""
    loader = _data_loader(dataset, config, shuffle=False, generator=None)
    feature_batches: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Cache ResNet features",
            unit="batch",
            leave=False,
        ):
            visual_features = model.encode_visual(
                batch["front"].to(device),
                batch["wrist"].to(device),
            )
            feature_batches.append(visual_features.detach().to("cpu", torch.float32))
    if not feature_batches:
        raise ValueError("Cannot cache image features from an empty dataset")
    return CachedImageBCDataset(dataset, torch.cat(feature_batches, dim=0))


def _train_epoch(
    model: StateBC | ImageBC,
    policy_type: BCPolicyType,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Train one epoch using L1 action regression and gradient clipping."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc="Train batches", unit="batch", leave=False)
    for batch in progress:
        prediction, action = _predict_batch(model, policy_type, batch, device)
        loss = nn.functional.l1_loss(prediction, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        batch_size = action.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        progress.set_postfix(l1=f"{float(loss.detach()):.4f}")
    if total_samples == 0:
        raise ValueError("Training dataloader produced no samples")
    return total_loss / total_samples


def _evaluate(
    model: StateBC | ImageBC,
    policy_type: BCPolicyType,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> BCValidationMetrics:
    """Compute L1, per-action MAE, and per-episode error without training updates."""
    model.eval()
    total_absolute_error = 0.0
    total_samples = 0
    action_error_totals = torch.zeros(len(ACTION_COMPONENT_NAMES), dtype=torch.float64)
    episode_error_totals: dict[str, float] = {}
    episode_sample_counts: dict[str, int] = {}
    with torch.inference_mode():
        for batch in loader:
            prediction, action = _predict_batch(model, policy_type, batch, device)
            absolute_error = (prediction - action).abs()
            batch_size = action.shape[0]
            total_absolute_error += float(absolute_error.sum())
            total_samples += batch_size
            action_error_totals += absolute_error.sum(dim=0).cpu().double()
            episode_indices = batch.get("episode_index")
            if episode_indices is not None:
                sample_errors = absolute_error.mean(dim=1).cpu().tolist()
                for episode_index, sample_error in zip(
                    episode_indices.tolist(), sample_errors, strict=True
                ):
                    key = str(int(episode_index))
                    episode_error_totals[key] = (
                        episode_error_totals.get(key, 0.0) + sample_error
                    )
                    episode_sample_counts[key] = episode_sample_counts.get(key, 0) + 1
    if total_samples == 0:
        raise ValueError("Validation dataloader produced no samples")
    action_mae = (action_error_totals / total_samples).tolist()
    return BCValidationMetrics(
        l1=total_absolute_error / (total_samples * len(ACTION_COMPONENT_NAMES)),
        action_mae=dict(zip(ACTION_COMPONENT_NAMES, action_mae, strict=True)),
        episode_l1={
            key: episode_error_totals[key] / episode_sample_counts[key]
            for key in sorted(episode_error_totals, key=int)
        },
    )


def _copy_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Copy state tensors to CPU so best-validation weights survive later epochs."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
