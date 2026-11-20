"""Decode LeRobot episodes once into a 5 Hz frame cache and serve JEPA windows."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional
from torch.utils.data import Dataset
from tqdm import tqdm

from common.seeding import named_generators
from envs.pick_place_env import CUBE_COLORS
from motion.control import WORKSPACE_LOWER, WORKSPACE_UPPER

FRAME_STRIDE = 4
FRONT_SIZE = 160
WRIST_SIZE = 112
ACTION_DIMENSION = 5
ACTION_BLOCK_DIMENSION = FRAME_STRIDE * ACTION_DIMENSION
CACHE_VERSION = 1
CAMERA_KEYS = {
    "front": "observation.images.front",
    "wrist": "observation.images.wrist",
}
_NUMERIC_COLUMNS = (
    "observation.state",
    "observation.environment_state",
    "action.executed",
    "episode_index",
    "frame_index",
)
_COLOR_PATTERN = re.compile(rf"\b({'|'.join(CUBE_COLORS)}) cube\b")
_VIDEO_TOLERANCE_SECONDS = 1e-4

DatasetKind = Literal["interaction", "demonstration"]


class CachedEpisode(BaseModel):
    """Locate one episode's 5 Hz timesteps in a frame cache, with provenance."""

    model_config = ConfigDict(extra="forbid")

    episode_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    source: str
    task_label: str
    target_object: str
    start: int = Field(ge=0)
    length: int = Field(ge=1)
    frame_count: int = Field(ge=1)


class FrameCacheKey(BaseModel):
    """Identify the decoded data so a stale cache is never reused."""

    model_config = ConfigDict(extra="forbid")

    cache_version: int = CACHE_VERSION
    dataset_id: str
    dataset_version: str
    collection_sha256: str
    info_sha256: str
    split: str | None
    tasks: list[str]
    max_episodes: int | None = None
    stride: int = FRAME_STRIDE
    front_size: int = FRONT_SIZE
    wrist_size: int = WRIST_SIZE


class FrameCacheManifest(BaseModel):
    """Describe a frame cache's key, episodes, and build cost."""

    model_config = ConfigDict(extra="forbid")

    key: FrameCacheKey
    episodes: list[CachedEpisode]
    total_timesteps: int = Field(ge=1)
    build_seconds: float = Field(ge=0.0)


@dataclass(frozen=True)
class EpisodeSource:
    """Describe one LeRobot episode to decode into the cache."""

    episode_index: int
    seed: int
    source: str


class FrameCache:
    """Read a built frame cache: images memory-mapped, labels and actions in memory."""

    def __init__(self, directory: str | Path) -> None:
        """Open a cache directory produced by ``build_frame_cache``."""
        self._open(Path(directory))

    def _open(self, directory: Path) -> None:
        """Load the manifest, map the image arrays, and validate every shape."""
        self.directory = directory.expanduser().resolve()
        manifest_path = self.directory / "cache.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Frame cache manifest is missing: {manifest_path}")
        self.manifest = FrameCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        self.front: NDArray[np.uint8] = np.load(
            self.directory / "front.npy", mmap_mode="r"
        )
        self.wrist: NDArray[np.uint8] = np.load(
            self.directory / "wrist.npy", mmap_mode="r"
        )
        self.action_blocks: NDArray[np.float32] = np.load(
            self.directory / "action_blocks.npy"
        )
        self.state: NDArray[np.float32] = np.load(self.directory / "state.npy")
        self.environment_state: NDArray[np.float32] = np.load(
            self.directory / "environment_state.npy"
        )
        total = self.manifest.total_timesteps
        expected_shapes = {
            "front": (total, FRONT_SIZE, FRONT_SIZE, 3),
            "wrist": (total, WRIST_SIZE, WRIST_SIZE, 3),
            "action_blocks": (total, FRAME_STRIDE, ACTION_DIMENSION),
            "state": (total, 12),
            "environment_state": (total, 28),
        }
        for name, shape in expected_shapes.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(
                    f"Frame cache array {name} has shape {actual}, expected {shape}"
                )
        covered = sum(episode.length for episode in self.manifest.episodes)
        if covered != total:
            raise ValueError("Frame cache episode table does not cover every timestep")

    @property
    def episodes(self) -> list[CachedEpisode]:
        """Return the cached episodes in storage order."""
        return self.manifest.episodes

    def __getstate__(self) -> dict[str, Any]:
        """Pickle only the directory, so worker processes reopen the memory maps."""
        return {"directory": self.directory}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reopen the cache in a worker process."""
        self._open(Path(state["directory"]))

    @property
    def dataset_id(self) -> str:
        """Return the source dataset identifier."""
        return self.manifest.key.dataset_id

    def episode_of_row(self) -> NDArray[np.int64]:
        """Return, for every storage row, the position of its episode in ``episodes``."""
        lengths = [episode.length for episode in self.episodes]
        return np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)


def frame_cache_name(key: FrameCacheKey) -> str:
    """Return a readable, content-addressed directory name for a cache key."""
    serialized = json.dumps(key.model_dump(mode="json"), sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    stem = key.dataset_id.removeprefix("blindspot/")
    split = f"-{key.split}" if key.split else ""
    return f"{stem}{split}-{digest}"


def resize_frames(frames: Tensor | NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Resize uint8 frames, channels-first or -last, to (N, size, size, 3) with antialiasing."""
    tensor = torch.as_tensor(frames)
    if tensor.dtype != torch.uint8 or tensor.ndim != 4:
        raise ValueError("Frames must be a uint8 array with four dimensions")
    if tensor.shape[1] != 3:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] != 3:
        raise ValueError("Frames must have three color channels")
    if tuple(tensor.shape[2:]) == (size, size):
        resized = tensor
    else:
        resized = (
            functional.interpolate(
                tensor.to(torch.float32),
                size=(size, size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
    return resized.permute(0, 2, 3, 1).contiguous().numpy()


def timesteps_for_frames(frame_count: int, stride: int = FRAME_STRIDE) -> int:
    """Return how many frames are kept when keeping every ``stride``-th frame."""
    return (frame_count + stride - 1) // stride


def action_blocks_for_episode(
    executed_actions: NDArray[np.floating], stride: int = FRAME_STRIDE
) -> NDArray[np.float32]:
    """Group the ``stride`` executed actions after each kept frame; NaN past the end."""
    actions = np.asarray(executed_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIMENSION:
        raise ValueError("Executed actions must have shape (frames, 5)")
    timesteps = timesteps_for_frames(len(actions), stride)
    blocks = np.full((timesteps, stride, ACTION_DIMENSION), np.nan, dtype=np.float32)
    for timestep in range(timesteps):
        chunk = actions[timestep * stride : (timestep + 1) * stride]
        blocks[timestep, : len(chunk)] = chunk
    return blocks


def normalize_action_blocks(blocks: NDArray[np.floating] | Tensor) -> Tensor:
    """Map raw action blocks (..., 4, 5) to the flat normalized 20-D JEPA input."""
    tensor = torch.as_tensor(blocks, dtype=torch.float32)
    if tuple(tensor.shape[-2:]) != (FRAME_STRIDE, ACTION_DIMENSION):
        raise ValueError("Action blocks must end with shape (4, 5)")
    lower = torch.as_tensor(WORKSPACE_LOWER, dtype=torch.float32)
    upper = torch.as_tensor(WORKSPACE_UPPER, dtype=torch.float32)
    position = 2.0 * (tensor[..., :3] - lower) / (upper - lower) - 1.0
    yaw = tensor[..., 3:4] / (np.pi / 2)
    gripper = 2.0 * tensor[..., 4:5] - 1.0
    return torch.cat((position, yaw, gripper), dim=-1).flatten(-2)


def denormalize_action_blocks(normalized: Tensor) -> Tensor:
    """Invert ``normalize_action_blocks`` back to raw (..., 4, 5) actions."""
    if normalized.shape[-1] != ACTION_BLOCK_DIMENSION:
        raise ValueError("Normalized action blocks must have 20 features")
    values = normalized.unflatten(-1, (FRAME_STRIDE, ACTION_DIMENSION))
    lower = torch.as_tensor(WORKSPACE_LOWER, dtype=values.dtype)
    upper = torch.as_tensor(WORKSPACE_UPPER, dtype=values.dtype)
    position = (values[..., :3] + 1.0) / 2.0 * (upper - lower) + lower
    yaw = values[..., 3:4] * (np.pi / 2)
    gripper = (values[..., 4:5] + 1.0) / 2.0
    return torch.cat((position, yaw, gripper), dim=-1)


def target_object_from_task(task_label: str) -> str:
    """Return the named cube body (for example ``cube_red``) from a task string."""
    match = _COLOR_PATTERN.search(task_label.lower())
    if match is None:
        raise ValueError(f"Task label names no cube color: {task_label!r}")
    return f"cube_{match.group(1)}"


def dataset_episode_sources(
    dataset_dir: Path,
    *,
    tasks: Sequence[str],
) -> tuple[DatasetKind, str | None, list[EpisodeSource]]:
    """Read a collection manifest and list its episodes for the requested tasks.

    Interaction corpora keep every episode in one LeRobot root; demonstration
    datasets contribute only their training split.
    """
    manifest_path = dataset_dir / "collection.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing collection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "episodes" in manifest and "source_ids" in manifest:
        sources = [
            EpisodeSource(
                int(record["episode_index"]), int(record["seed"]), str(record["source"])
            )
            for record in manifest["episodes"]
            if record["task"] in tasks
        ]
        return "interaction", None, sources
    task = manifest.get("task")
    if task not in tasks:
        raise ValueError(
            f"Dataset task {task!r} is not in the requested tasks {list(tasks)}"
        )
    sources = [
        EpisodeSource(index, int(seed), "demonstration")
        for index, seed in enumerate(manifest["train"]["seeds"])
    ]
    return "demonstration", "train", sources


def _file_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frame_cache(
    dataset_dir: str | Path,
    cache_root: str | Path,
    *,
    tasks: Sequence[str] = ("T3",),
    max_episodes: int | None = None,
) -> FrameCache:
    """Decode a dataset's episodes at 5 Hz into a cache, reusing a matching one."""
    dataset_path = Path(dataset_dir).expanduser().resolve()
    _, split, sources = dataset_episode_sources(dataset_path, tasks=tasks)
    if max_episodes is not None:
        sources = sources[:max_episodes]
    if not sources:
        raise ValueError(f"No {list(tasks)} episodes found in {dataset_path}")
    lerobot_root = dataset_path / split if split else dataset_path
    dataset_id = f"blindspot/{dataset_path.name}"
    repo_id = f"{dataset_id}-{split}" if split else dataset_id
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.video_utils import decode_video_frames
    except ImportError as error:
        raise RuntimeError(
            "Building a frame cache requires LeRobot; run `uv sync --extra jepa`."
        ) from error
    dataset = LeRobotDataset(repo_id=repo_id, root=lerobot_root, video_backend="pyav")
    key = FrameCacheKey(
        dataset_id=dataset_id,
        dataset_version=str(dataset.meta.info.codebase_version),
        collection_sha256=_file_sha256(dataset_path / "collection.json"),
        info_sha256=_file_sha256(lerobot_root / "meta" / "info.json"),
        split=split,
        tasks=sorted(tasks),
        max_episodes=max_episodes,
    )
    directory = Path(cache_root).expanduser().resolve() / frame_cache_name(key)
    if (directory / "cache.json").is_file():
        cache = FrameCache(directory)
        if cache.manifest.key != key:
            raise ValueError(f"Frame cache {directory} exists with a different key")
        return cache

    numeric = dataset.hf_dataset.select_columns(list(_NUMERIC_COLUMNS)).with_format(
        "numpy"
    )

    def decode(episode: EpisodeSource) -> dict[str, Any]:
        """Decode one episode's kept frames, labels, and action blocks."""
        record = dataset.meta.episodes[episode.episode_index]
        start = int(record["dataset_from_index"])
        stop = int(record["dataset_to_index"])
        rows = numeric[start:stop]
        if np.any(rows["episode_index"] != episode.episode_index) or not np.array_equal(
            rows["frame_index"], np.arange(stop - start)
        ):
            raise ValueError(f"Episode {episode.episode_index} rows are not contiguous")
        kept = np.arange(0, stop - start, FRAME_STRIDE)
        images = {}
        for camera, size in (("front", FRONT_SIZE), ("wrist", WRIST_SIZE)):
            video_key = CAMERA_KEYS[camera]
            offset = float(record[f"videos/{video_key}/from_timestamp"])
            video_path = lerobot_root / dataset.meta.get_video_file_path(
                episode.episode_index, video_key
            )
            frames = decode_video_frames(
                video_path,
                [offset + float(index) / dataset.fps for index in kept],
                _VIDEO_TOLERANCE_SECONDS,
                "pyav",
                return_uint8=True,
            )
            if frames.shape[0] != len(kept):
                raise ValueError(
                    f"Decoded {frames.shape[0]}/{len(kept)} frames for {video_key}"
                )
            images[camera] = resize_frames(frames, size)
        return {
            "front": images["front"],
            "wrist": images["wrist"],
            "action_blocks": action_blocks_for_episode(rows["action.executed"]),
            "state": np.asarray(rows["observation.state"][kept], dtype=np.float32),
            "environment_state": np.asarray(
                rows["observation.environment_state"][kept], dtype=np.float32
            ),
            "task_label": str(record["tasks"][0]),
            "frame_count": stop - start,
        }

    timesteps = [
        timesteps_for_frames(int(dataset.meta.episodes[source.episode_index]["length"]))
        for source in sources
    ]
    return write_frame_cache(directory, key, sources, timesteps, decode)


def write_frame_cache(
    directory: Path,
    key: FrameCacheKey,
    sources: Sequence[EpisodeSource],
    timesteps: Sequence[int],
    decode: Callable[[EpisodeSource], dict[str, Any]],
) -> FrameCache:
    """Stream every decoded episode into pre-sized arrays and publish the cache."""
    if len(sources) != len(timesteps) or any(count < 1 for count in timesteps):
        raise ValueError("Every episode needs a positive timestep count")
    started = time.perf_counter()
    total = int(sum(timesteps))
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        layouts = {
            "front": ((total, FRONT_SIZE, FRONT_SIZE, 3), np.uint8),
            "wrist": ((total, WRIST_SIZE, WRIST_SIZE, 3), np.uint8),
            "action_blocks": ((total, FRAME_STRIDE, ACTION_DIMENSION), np.float32),
            "state": ((total, 12), np.float32),
            "environment_state": ((total, 28), np.float32),
        }
        arrays = {
            name: np.lib.format.open_memmap(
                staging / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
            )
            for name, (shape, dtype) in layouts.items()
        }
        episodes: list[CachedEpisode] = []
        start = 0
        for source, length in tqdm(
            list(zip(sources, timesteps, strict=True)),
            desc=f"Decoding {key.dataset_id}",
            unit="episode",
        ):
            values = decode(source)
            for name, array in arrays.items():
                if len(values[name]) != length:
                    raise ValueError(
                        f"Episode {source.episode_index} {name} has "
                        f"{len(values[name])} rows, expected {length}"
                    )
                array[start : start + length] = values[name]
            episodes.append(
                CachedEpisode(
                    episode_index=source.episode_index,
                    seed=source.seed,
                    source=source.source,
                    task_label=values["task_label"],
                    target_object=target_object_from_task(values["task_label"]),
                    start=start,
                    length=length,
                    frame_count=int(values["frame_count"]),
                )
            )
            start += length
        for array in arrays.values():
            array.flush()
        del arrays
        manifest = FrameCacheManifest(
            key=key,
            episodes=episodes,
            total_timesteps=total,
            build_seconds=time.perf_counter() - started,
        )
        (staging / "cache.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return FrameCache(directory)


@dataclass(frozen=True)
class EpisodeRef:
    """Point to one episode within one of several caches."""

    cache_index: int
    episode: CachedEpisode


def all_episodes(caches: Sequence[FrameCache]) -> list[EpisodeRef]:
    """Return every episode of every cache in storage order."""
    return [
        EpisodeRef(cache_index, episode)
        for cache_index, cache in enumerate(caches)
        for episode in cache.episodes
    ]


def split_holdout_episodes(
    caches: Sequence[FrameCache],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    """Hold out a source-stratified fraction of episodes, reproducibly."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must lie in [0, 1)")
    by_source: dict[str, list[EpisodeRef]] = {}
    for reference in all_episodes(caches):
        by_source.setdefault(reference.episode.source, []).append(reference)
    random = named_generators(seed, ("holdout",))["holdout"]
    train: list[EpisodeRef] = []
    held_out: list[EpisodeRef] = []
    for source in sorted(by_source):
        references = by_source[source]
        held = set(
            random.permutation(len(references))[
                : round(fraction * len(references))
            ].tolist()
        )
        for index, reference in enumerate(references):
            (held_out if index in held else train).append(reference)
    return train, held_out


def window_starts(episodes: Iterable[EpisodeRef], window: int) -> NDArray[np.int64]:
    """Return (cache index, first row) for every window inside a single episode."""
    if window < 1:
        raise ValueError("window must be positive")
    starts = [
        (reference.cache_index, reference.episode.start + offset)
        for reference in episodes
        for offset in range(reference.episode.length - window + 1)
    ]
    return np.asarray(starts, dtype=np.int64).reshape(-1, 2)


class WindowDataset(Dataset[dict[str, Tensor]]):
    """Serve windows of both cameras and the action blocks between their timesteps."""

    def __init__(
        self,
        caches: Sequence[FrameCache],
        episodes: Sequence[EpisodeRef],
        *,
        window: int = 4,
    ) -> None:
        """Index every window of ``window`` timesteps that stays inside one episode."""
        self.caches = list(caches)
        self.window = window
        self.starts = window_starts(episodes, window)
        if len(self.starts) == 0:
            raise ValueError("No episode is long enough for one window")

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return uint8 images (T, H, W, 3), normalized action blocks (T-1, 20), and rows."""
        cache_index, start = (int(value) for value in self.starts[index])
        cache = self.caches[cache_index]
        rows = slice(start, start + self.window)
        blocks = cache.action_blocks[start : start + self.window - 1]
        if not np.all(np.isfinite(blocks)):
            raise ValueError(f"Window at row {start} has incomplete action blocks")
        return {
            "front": torch.from_numpy(np.array(cache.front[rows])),
            "wrist": torch.from_numpy(np.array(cache.wrist[rows])),
            "actions": normalize_action_blocks(blocks),
            "cache_index": torch.tensor(cache_index, dtype=torch.int64),
            "row": torch.tensor(start, dtype=torch.int64),
        }


class TimestepDataset(Dataset[dict[str, Tensor]]):
    """Serve single 5 Hz timesteps for supervised training and feature extraction."""

    def __init__(
        self, caches: Sequence[FrameCache], episodes: Sequence[EpisodeRef]
    ) -> None:
        """Index every timestep of the given episodes."""
        self.caches = list(caches)
        self.rows = window_starts(episodes, 1)

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one timestep's uint8 images and its location in the caches."""
        cache_index, row = (int(value) for value in self.rows[index])
        cache = self.caches[cache_index]
        return {
            "front": torch.from_numpy(np.array(cache.front[row])),
            "wrist": torch.from_numpy(np.array(cache.wrist[row])),
            "cache_index": torch.tensor(cache_index, dtype=torch.int64),
            "row": torch.tensor(row, dtype=torch.int64),
        }


def main() -> None:
    """Build (or reuse) the frame cache of each named dataset."""
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument(
        "--data-root", type=Path, default=project_root / "data" / "datasets"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=project_root / "data" / "cache" / "frames"
    )
    parser.add_argument("--tasks", nargs="+", default=["T3"])
    arguments = parser.parse_args()
    for name in arguments.datasets:
        cache = build_frame_cache(
            arguments.data_root / name, arguments.cache_root, tasks=arguments.tasks
        )
        print(
            f"{cache.directory}: {len(cache.episodes)} episodes, "
            f"{cache.manifest.total_timesteps} timesteps"
        )


if __name__ == "__main__":
    main()
