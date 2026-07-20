"""因果推断层 — Pearl do-calculus for visual materials science.

把材料科学图像特征 + 实验条件建模为 SCM, 支持 L2 干预预测.

模块:
  - visual_scm:          VisualSCM 表征 + 4 领域模板 (sintering/ostwald/diffusion/phase)
  - predict_intervention: L2 do-calculus 工具 (Monte Carlo 拓扑序采样)
  - llm_generate_scm:    LLM 生成 SCM 草稿 (KB 无模板时 fallback)
  - visual_causal_chain: Phase 2 从多图/多观测点自动拟合 SCM
  - counterfactual_render: Phase 3 L3 反事实渲染 (Monte Carlo abduction + prediction)

阶梯映射 (Pearl Causal Hierarchy):
  L1 观察  P(Y|X)           — vision_describe (感知层)
  L2 干预  P(Y|do(X))       — predict_intervention (Phase 1)
  L2+ 拟合 P(Y|do(X), data) — visual_causal_chain (Phase 2): 数据后验修正先验参数
  L3 反事实 P(Y_x|X',Y')    — counterfactual_render (Phase 3)

设计原则 (ponytail):
  - 节点是数学对象, 不是图像 (符合 physics=mathematics 偏好)
  - 物理先验优先 (Arrhenius/Ostwald/Fick/Avrami), 数据后验修正
  - 零新依赖 (不用 pgmpy/causalnex/doWhy)
  - LLM 生成 SCM 必须标 confirmed=False, predict 时显式警告

升级路径:
  - 方程可逆时用解析 abduction (替代 Monte Carlo rejection sampling)
  - 多节点 evidence 用 importance sampling / MCMC
  - 反事实渲染成图像 (visualize cf prediction as plot)
"""

from huginn.causal.visual_scm import (
    VisualSCM, Variable, Edge,
    list_templates, get_template, match_template,
)
from huginn.causal.predict_intervention import (
    predict_intervention, PredictInterventionTool,
)
from huginn.causal.visual_causal_chain import (
    Observation, fit_scm_from_observations,
    extract_observations_from_images,
    FitSCMFromObservationsTool,
)
from huginn.causal.counterfactual_render import (
    counterfactual_render, CounterfactualRenderTool,
)

__all__ = [
    "VisualSCM", "Variable", "Edge",
    "list_templates", "get_template", "match_template",
    "predict_intervention", "PredictInterventionTool",
    "Observation", "fit_scm_from_observations",
    "extract_observations_from_images",
    "FitSCMFromObservationsTool",
    "counterfactual_render", "CounterfactualRenderTool",
]
