"""HPC resource selection helpers.

Maps high-level execution hints (GPU, memory, queue profile) to concrete
scheduler queues and GPU counts based on ``HPCConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from huginn.hpc.client import HPCConfig


@dataclass
class ResourceSelection:
    """Concrete resources chosen for a remote job."""

    queue: str | None
    gpus_per_node: int
    profile: str


class ResourceSelector:
    """Select queue and GPU resources for an HPC job."""

    def __init__(self, config: HPCConfig) -> None:
        self.config = config

    def select(
        self,
        queue: str | None = None,
        gpu: bool | int | None = None,
        profile: str | None = None,
        gpus_per_node: int | None = None,
        **kwargs: Any,
    ) -> ResourceSelection:
        """Choose queue and GPU count from explicit hints and config defaults.

        Args:
            queue: Explicit queue/partition name. Always wins when provided.
            gpu: ``True`` to request the default GPU count, or an integer.
            profile: Named profile (e.g. ``"cpu"``, ``"gpu"``, ``"fat"``).
            gpus_per_node: Override GPU count directly.
            **kwargs: Ignored; accepts extra scheduler hints.
        """
        if queue:
            selected_queue = queue
        elif profile and profile in self.config.queue_map:
            selected_queue = self.config.queue_map[profile]
        elif gpu and self.config.gpu_queue:
            selected_queue = self.config.gpu_queue
        else:
            selected_queue = self.config.default_queue

        if gpus_per_node is not None:
            selected_gpus = int(gpus_per_node)
        elif gpu is True:
            selected_gpus = max(1, self.config.default_gpus_per_node)
        elif isinstance(gpu, int):
            selected_gpus = max(0, gpu)
        else:
            selected_gpus = self.config.default_gpus_per_node

        selected_profile = profile or ("gpu" if selected_gpus > 0 else "cpu")
        return ResourceSelection(
            queue=selected_queue,
            gpus_per_node=selected_gpus,
            profile=selected_profile,
        )
