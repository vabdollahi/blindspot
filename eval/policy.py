"""Shared protocol for policies consumed by the evaluation harness."""

from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import ArrayLike, NDArray


class Policy(Protocol):
    """Interface implemented by scripted and learned evaluation policies."""

    def reset(self, instruction: str | None = None) -> None:
        """Reset policy state before a new episode."""
        ...

    def act(self, obs: Mapping[str, Any]) -> ArrayLike | NDArray[Any]:
        """Return one action for the current observation."""
        ...
