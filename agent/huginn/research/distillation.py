"""蒸馏/教师产物的权威分类与晋升门（spec §18.6 / §19.5，I30）。

三轴分开：
- ``abstraction_authority``：能提供多抽象的框架；
- ``evidential_authority``：能作为多强的证据；
- ``influence_authority``：此刻能多大程度塑造推理（逐查询）。

规则：教师/蒸馏/聚类摘要默认 ``derived``；未经晋升不得作为直接证据；
晋升必须 provenance 完整 + ≥1 独立在线佐证 + 无未解矛盾。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

AuthorityClass = Literal["direct", "derived", "hypothesis"]


@dataclass(frozen=True)
class AuthorityVector:
    abstraction_authority: float
    evidential_authority: float
    influence_authority: float


_DEFAULT_VECTORS: dict[str, AuthorityVector] = {
    "direct": AuthorityVector(0.2, 0.9, 1.0),
    "derived": AuthorityVector(0.7, 0.4, 0.6),
    "hypothesis": AuthorityVector(0.9, 0.1, 0.4),
}


def authority_vector(source_class: AuthorityClass) -> AuthorityVector:
    return _DEFAULT_VECTORS.get(str(source_class), _DEFAULT_VECTORS["hypothesis"])


@dataclass(frozen=True)
class AuthorityLink:
    node_id: str
    source_class: AuthorityClass
    directness: float
    corroboration: int = 0
    provenance_digest: str = ""
    epistemic_status: str = "accepted"
    parent: str | None = None


def distinguish_source(source: str) -> AuthorityClass:
    """把生产来源映射到权威类别；未知/教师产物一律 derived（宁低勿高）."""
    text = str(source or "").strip().lower()
    if text.startswith(("tool_run", "exec:", "measure:", "replay:", "pybullet_run")):
        return "direct"
    if text.startswith(("distiller:", "teacher:", "cluster:", "dream:")):
        return "derived"
    if text.startswith(("hypothesis:", "speculation:", "imagination:")):
        return "hypothesis"
    return "derived"


def promote(
    link: AuthorityLink,
    *,
    provenance_complete: bool,
    independent_corroboration: int = 0,
    has_unresolved_contradiction: bool = False,
) -> AuthorityLink:
    """写入侧晋升门：hypothesis → derived → direct；矛盾则降为 contested."""
    if has_unresolved_contradiction:
        return replace(link, epistemic_status="contested", source_class="hypothesis")
    if link.source_class == "hypothesis":
        if provenance_complete and independent_corroboration >= 1:
            return replace(link, source_class="derived", corroboration=independent_corroboration)
        return link
    if link.source_class == "derived":
        if provenance_complete and independent_corroboration >= 1:
            return replace(link, source_class="direct", corroboration=independent_corroboration)
        return link
    return link


def can_be_direct_evidence(link: AuthorityLink) -> bool:
    return (
        link.source_class == "direct"
        and link.provenance_digest != ""
        and link.epistemic_status == "accepted"
    )


__all__ = [
    "AuthorityClass",
    "AuthorityLink",
    "AuthorityVector",
    "authority_vector",
    "can_be_direct_evidence",
    "distinguish_source",
    "promote",
]
