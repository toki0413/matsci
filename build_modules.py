"""Extract verbatim function bodies from rcb_runner.py into new modules."""
SRC = "/workspace/agent/huginn/cli/rcb_runner.py"
with open(SRC, encoding="utf-8") as f:
    lines = f.readlines()


def grab(a, b):
    """lines[a-1:b] verbatim (1-indexed inclusive)."""
    return "".join(lines[a - 1:b])


# ---------- rcb_step2.py ----------
step2 = [
    '"""RCB Step 2 执行 — 依赖 rcb_utils / rcb_cognition / rcb_audit."""\n',
    "from __future__ import annotations\n",
    "\n",
    "import contextlib\n",
    "import logging\n",
    "import os\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "from typing import Any\n",
    "\n",
    "from huginn.cli.rcb.audit import _rcb_drift_check\n",
    "from huginn.cli.rcb.prompt_builders import _legacy_build_step2_prompt\n",
    "from huginn.cli.rcb_audit import (\n",
    "    _ChecklistItem,\n",
    "    _checklist_item_parser,\n",
    "    _derivation_chain_audit,\n",
    "    _llm_coverage_audit,\n",
    "    _rcb_effort_floor,\n",
    "    _report_coverage_compass,\n",
    "    _time_slot_index,\n",
    ")\n",
    "from huginn.cli.rcb_fork_merge import (\n",
    "    _FCM_PERSPECTIVES,\n",
    "    anneal_fork_count,\n",
    "    judge_fork_reports,\n",
    ")\n",
    "from huginn.cli.rcb_cognition import (\n",
    "    _append_observations_log,\n",
    "    _collect_observations,\n",
    "    _compute_v15_fields,\n",
    "    _init_hypothesis_manifold,\n",
    "    _record_abduction,\n",
    "    _trigger_anomaly_hypothesis,\n",
    "    _write_cognitive_evidence,\n",
    ")\n",
    "from huginn.cli.rcb_utils import (\n",
    "    _detect_file_rewrite_stagnation,\n",
    "    _infer_domain,\n",
    "    _infer_task_id_from_workspace,\n",
    "    _make_simplex_id,\n",
    "    _MODEL_VERSION,\n",
    "    _save_manifold,\n",
    "    _cross_task_store,\n",
    ")\n",
    "from huginn.utils.runtime import HUGINN_DIR_NAME, get_runtime_home\n",
    "\n",
    "logger = logging.getLogger(__name__)\n",
    "\n",
    "\n",
    grab(766, 793),
    "\n",
    "\n",
    grab(796, 2879),
]

with open("/workspace/agent/huginn/cli/rcb_step2.py", "w", encoding="utf-8") as f:
    f.writelines(step2)

# ---------- rcb_step3.py ----------
step3 = [
    '"""RCB Step 3 对抗审计 — 依赖 rcb_utils / rcb_critique / rcb_fork_merge."""\n',
    "from __future__ import annotations\n",
    "\n",
    "import contextlib\n",
    "import json\n",
    "import logging\n",
    "import os\n",
    "import time\n",
    "from pathlib import Path\n",
    "from typing import Any\n",
    "\n",
    "from huginn.cli.rcb.audit import (\n",
    "    _derive_gap_type,\n",
    "    _infer_beta_1_simple,\n",
    "    _recompute_report_metrics,\n",
    "    _should_retry_execute,\n",
    "    _write_directive_rejection,\n",
    ")\n",
    "from huginn.cli.rcb_critique import adversarial_critique, format_critique_for_agent\n",
    "from huginn.cli.rcb_fork_merge import _reproduction_gate\n",
    "from huginn.cli.rcb_utils import (\n",
    "    _infer_domain,\n",
    "    _infer_task_id_from_workspace,\n",
    "    _make_simplex_id,\n",
    "    _MODEL_VERSION,\n",
    ")\n",
    "from huginn.utils.runtime import HUGINN_DIR_NAME\n",
    "\n",
    "logger = logging.getLogger(__name__)\n",
    "\n",
    "\n",
    grab(3304, 3852),
]

with open("/workspace/agent/huginn/cli/rcb_step3.py", "w", encoding="utf-8") as f:
    f.writelines(step3)

# ---------- rcb_mcmc.py ----------
mcmc = [
    '"""RCB MCMC 模式 — 依赖 rcb_cognition / rcb_utils."""\n',
    "from __future__ import annotations\n",
    "\n",
    "import contextlib\n",
    "import json\n",
    "import logging\n",
    "import os\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "from huginn.cli.rcb_cognition import _init_hypothesis_manifold\n",
    "from huginn.utils.runtime import HUGINN_DIR_NAME\n",
    "\n",
    "logger = logging.getLogger(__name__)\n",
    "\n",
    "\n",
    grab(4958, 5305),
]

with open("/workspace/agent/huginn/cli/rcb_mcmc.py", "w", encoding="utf-8") as f:
    f.writelines(mcmc)

print("done")