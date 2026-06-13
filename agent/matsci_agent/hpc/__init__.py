"""HPC remote execution module.

Supports SLURM and PBS job submission via SSH.
"""

from matsci_agent.hpc.client import HPCClient, HPCConfig

__all__ = ["HPCClient", "HPCConfig"]
