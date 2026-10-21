"""I30: 教师/蒸馏产物为 derived；未经晋升不得作为直接证据."""
from __future__ import annotations

from huginn.research.distillation import (
    AuthorityLink,
    authority_vector,
    can_be_direct_evidence,
    distinguish_source,
    promote,
)


def test_derived_only():
    source = distinguish_source("distiller:lesson-1")
    assert source == "derived"
    vector = authority_vector(source)
    assert vector.evidential_authority <= 0.6
    assert vector.evidential_authority < authority_vector("direct").evidential_authority

    link = AuthorityLink(
        node_id="k1", source_class="derived", directness=1.0,
        provenance_digest="sha:abc",
    )
    assert can_be_direct_evidence(link) is False, "derived 不能直接当证据"

    # 未满足晋升条件 → 保持 derived
    assert promote(link, provenance_complete=True, independent_corroboration=0).source_class == "derived"
    # 满足条件 → 升 direct，且可作直接证据
    promoted = promote(
        link, provenance_complete=True, independent_corroboration=2
    )
    assert promoted.source_class == "direct"
    assert can_be_direct_evidence(promoted) is True

    # 有未解矛盾 → contested + hypothesis（保留为负证据）
    contested = promote(
        link, provenance_complete=True, independent_corroboration=2,
        has_unresolved_contradiction=True,
    )
    assert contested.epistemic_status == "contested"
    assert contested.source_class == "hypothesis"
    assert can_be_direct_evidence(contested) is False

    # 教师产物永远是 derived，即使 evidence 很多
    teacher = AuthorityLink(
        node_id="t1", source_class=distinguish_source("teacher:demo"),
        directness=1.0, provenance_digest="sha:def",
    )
    assert teacher.source_class == "derived"
