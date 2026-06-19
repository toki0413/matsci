"""Tests for HPC client script generation and resource selection."""

from __future__ import annotations

from huginn.hpc.client import HPCClient, HPCConfig
from huginn.hpc.resource_selector import ResourceSelector


class TestHPCScriptGeneration:
    def test_slurm_script_includes_gpu_directive(self):
        config = HPCConfig(host="hpc.example.com", username="user")
        client = HPCClient(config)
        script = client.generate_job_script(
            command="echo hello",
            job_name="gpu_test",
            queue="gpu",
            gpus_per_node=2,
        )
        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH --partition=gpu" in script

    def test_slurm_script_without_gpu_omits_gres(self):
        config = HPCConfig(host="hpc.example.com", username="user")
        client = HPCClient(config)
        script = client.generate_job_script(
            command="echo hello",
            job_name="cpu_test",
            queue="normal",
            gpus_per_node=0,
        )
        assert "--gres=gpu" not in script
        assert "#SBATCH --partition=normal" in script

    def test_pbs_script_includes_gpu_directive(self):
        config = HPCConfig(host="hpc.example.com", username="user", scheduler="pbs")
        client = HPCClient(config)
        script = client.generate_job_script(
            command="echo hello",
            job_name="gpu_test",
            queue="gpu",
            gpus_per_node=4,
        )
        assert ":ngpus=4" in script
        assert "#PBS -q gpu" in script


class TestResourceSelector:
    def test_explicit_queue_wins(self):
        config = HPCConfig(
            host="hpc.example.com",
            username="user",
            default_queue="normal",
            gpu_queue="gpu",
            queue_map={"fat": "fat_nodes"},
        )
        selector = ResourceSelector(config)
        selection = selector.select(queue="custom", profile="fat")
        assert selection.queue == "custom"

    def test_gpu_hint_routes_to_gpu_queue(self):
        config = HPCConfig(
            host="hpc.example.com",
            username="user",
            default_queue="normal",
            gpu_queue="gpu",
            default_gpus_per_node=0,
        )
        selector = ResourceSelector(config)
        selection = selector.select(gpu=True)
        assert selection.queue == "gpu"
        assert selection.gpus_per_node == 1
        assert selection.profile == "gpu"

    def test_profile_queue_map(self):
        config = HPCConfig(
            host="hpc.example.com",
            username="user",
            default_queue="normal",
            queue_map={"fat": "fat_nodes"},
        )
        selector = ResourceSelector(config)
        selection = selector.select(profile="fat")
        assert selection.queue == "fat_nodes"
        assert selection.gpus_per_node == 0

    def test_integer_gpu_count(self):
        config = HPCConfig(
            host="hpc.example.com",
            username="user",
            default_gpus_per_node=1,
        )
        selector = ResourceSelector(config)
        selection = selector.select(gpu=4)
        assert selection.gpus_per_node == 4

    def test_default_queue_when_no_hints(self):
        config = HPCConfig(
            host="hpc.example.com",
            username="user",
            default_queue="normal",
        )
        selector = ResourceSelector(config)
        selection = selector.select()
        assert selection.queue == "normal"
        assert selection.gpus_per_node == 0
