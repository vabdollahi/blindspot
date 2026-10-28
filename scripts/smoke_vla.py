"""Smoke-test a VLA checkpoint on two held-out T3-L color instructions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from data.vla_subset import (
    VLA_TARGET_COLORS,
    active_cube_colors_from_positions,
)
from envs.pick_place_env import PickPlaceEnv, get_instruction_templates
from eval.policy import Policy
from policies.vla import load_vla_policy

_VALIDATION_SEED_START = 100_000
_VALIDATION_SEED_STOP = 110_000


def _instruction_color(instruction: str) -> str:
    """Extract the cube color named by a configured T3-L template."""
    words = set(instruction.lower().replace(".", " ").split())
    for color in ("red", "blue", "green"):
        if color in words:
            return color
    raise ValueError(f"T3-L instruction does not name a supported color: {instruction}")


def _run_color_episode(
    environment: PickPlaceEnv,
    policy: Policy,
    *,
    seed: int,
    template: str,
    completed_colors: set[str],
    profile: Literal["full", "limited"],
) -> tuple[str, str, int, bool, bool, list[float]] | None:
    """Run one held-out-template episode and return outcome and action latencies."""
    observation, _ = environment.reset(
        seed=seed,
        options={"instruction_template": template},
    )
    instruction = observation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("T3-L environment did not return its language instruction")
    color = _instruction_color(instruction)
    object_poses = observation.get("object_poses")
    if not isinstance(object_poses, dict):
        raise TypeError("VLA smoke evaluation requires object poses for diagnostics")
    active_colors = active_cube_colors_from_positions(object_poses)
    if profile == "limited" and (
        color not in VLA_TARGET_COLORS or len(active_colors) != 2
    ):
        return None
    if color in completed_colors:
        return None
    other_initial_positions = {
        other_color: np.asarray(
            object_poses[f"cube_{other_color}"]["position"],
            dtype=np.float64,
        )
        for other_color in active_colors
        if other_color != color
    }
    wrong_object_moved = False
    observation.update(environment.observe_images())
    policy.reset(instruction)
    latencies: list[float] = []
    action = policy.act(observation)
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = 0
    while not terminated and not truncated:
        if steps > 0:
            started = time.perf_counter()
            action = policy.act(observation)
            latencies.append(time.perf_counter() - started)
        observation, _, terminated, truncated, info = environment.step(action)
        steps += 1
        next_object_poses = observation.get("object_poses")
        if not isinstance(next_object_poses, dict):
            raise TypeError("VLA smoke step did not return object poses")
        for other_color, initial_position in other_initial_positions.items():
            current_position = np.asarray(
                next_object_poses[f"cube_{other_color}"]["position"],
                dtype=np.float64,
            )
            if np.linalg.norm(current_position - initial_position) > 0.02:
                wrong_object_moved = True
        if not terminated and not truncated:
            observation.update(environment.observe_images())
    return (
        color,
        instruction,
        steps,
        bool(info.get("success", False)),
        wrong_object_moved,
        latencies,
    )


def main() -> None:
    """Run the same held-out paraphrase with two sampled cube colors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level", choices=("L1", "L2"), default="L1")
    parser.add_argument(
        "--profile",
        choices=("checkpoint", "full", "limited"),
        default="checkpoint",
        help="Default to the training profile stored in the checkpoint.",
    )
    parser.add_argument("--seed-start", type=int, default=_VALIDATION_SEED_START)
    parser.add_argument("--max-seeds", type=int, default=256)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.max_seeds < 2:
        parser.error("--max-seeds must be at least 2")
    if (
        arguments.seed_start < _VALIDATION_SEED_START
        or arguments.seed_start + arguments.max_seeds > _VALIDATION_SEED_STOP
    ):
        parser.error("smoke seeds must stay within the reserved validation range")
    templates = get_instruction_templates("T3-L", "held-out")
    if not templates:
        raise ValueError("T3-L has no held-out instruction templates")

    policy = load_vla_policy(arguments.checkpoint, device=arguments.device)
    profile = (
        policy.training_profile
        if arguments.profile == "checkpoint"
        else arguments.profile
    )
    environment = PickPlaceEnv(
        "T3-L",
        arguments.level,
        obs_mode="state",
        image_size=256,
    )
    completed_colors: set[str] = set()
    all_latencies: list[float] = []
    wrong_object_episodes = 0
    try:
        for seed in range(
            arguments.seed_start, arguments.seed_start + arguments.max_seeds
        ):
            episode = _run_color_episode(
                environment,
                policy,
                seed=seed,
                template=templates[0],
                completed_colors=completed_colors,
                profile=profile,
            )
            if episode is None:
                continue
            color, instruction, steps, success, wrong_object_moved, latencies = episode
            completed_colors.add(color)
            all_latencies.extend(latencies)
            wrong_object_episodes += int(wrong_object_moved)
            print(
                f"seed={seed} instruction={instruction!r} "
                f"steps={steps} success={success} "
                f"wrong_object_moved={wrong_object_moved}"
            )
            if len(completed_colors) == 2:
                break
    finally:
        environment.close()

    if len(completed_colors) != 2:
        raise RuntimeError(
            f"Found only {len(completed_colors)} distinct target colors in "
            f"{arguments.max_seeds} validation seeds"
        )
    if not all_latencies:
        raise RuntimeError("VLA smoke rollouts produced no timed policy actions")
    print(
        "policy_act_latency_ms "
        f"p50={np.percentile(all_latencies, 50) * 1000:.2f} "
        f"p95={np.percentile(all_latencies, 95) * 1000:.2f}"
    )
    print(f"wrong_object_episodes={wrong_object_episodes}/2")


if __name__ == "__main__":
    main()
