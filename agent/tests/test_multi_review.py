"""multi_review action 测试 — nuwa 6 路并行透镜 + cangjie 三重验证.

3 测:
  1. _triple_verify 纯逻辑: V1/V2/V3 打分 (no LLM, no network)
  2. 透镜失败降级: mock LLM 返回无效 JSON, 失败透镜进 lenses_failed, 不阻塞其他
  3. 无 papers 报错
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from huginn.tools.literature.tool import LiteratureInput, LiteratureTool


class TestTripleVerify:
    """_triple_verify 纯逻辑测试 — cangjie V1/V2/V3 打分."""

    def test_v1_cross_domain_two_papers(self):
        # claim 出现在 2 篇 → V1=True
        v1, _, _ = LiteratureTool._triple_verify(
            {"claim": "某方法有效", "paper_idx": [1, 2]}, n_papers=4
        )
        assert v1 is True

    def test_v1_single_paper_false(self):
        # claim 只出现在 1 篇 → V1=False
        v1, _, _ = LiteratureTool._triple_verify(
            {"claim": "某方法有效", "paper_idx": [1]}, n_papers=4
        )
        assert v1 is False

    def test_v2_generative_if_marker(self):
        # claim 含 "如果" → V2=True
        _, v2, _ = LiteratureTool._triple_verify(
            {"claim": "如果温度升高则反应加快", "paper_idx": [1]}, n_papers=3
        )
        assert v2 is True

    def test_v2_generative_predict_marker(self):
        # claim 含 "predict" → V2=True
        _, v2, _ = LiteratureTool._triple_verify(
            {"claim": "we predict this trend continues", "paper_idx": [1]}, n_papers=3
        )
        assert v2 is True

    def test_v2_no_generative_marker(self):
        # claim 纯描述, 无预测结构 → V2=False
        _, v2, _ = LiteratureTool._triple_verify(
            {"claim": "作者用了 DFT 计算", "paper_idx": [1, 2]}, n_papers=4
        )
        assert v2 is False

    def test_v3_exclusive_under_half(self):
        # claim 出现在 1/4 篇 → V3=True (排他)
        _, _, v3 = LiteratureTool._triple_verify(
            {"claim": "稀有发现", "paper_idx": [1]}, n_papers=4
        )
        assert v3 is True

    def test_v3_not_exclusive_at_half(self):
        # claim 出现在 2/4 篇 → V3=False (常识)
        _, _, v3 = LiteratureTool._triple_verify(
            {"claim": "常见结论", "paper_idx": [1, 2]}, n_papers=4
        )
        assert v3 is False

    def test_score_3_is_high(self):
        # V1+V2+V3 全 True → score=3 → high
        v1, v2, v3 = LiteratureTool._triple_verify(
            {"claim": "如果 X 则 Y (predict)", "paper_idx": [1, 2]}, n_papers=5
        )
        assert (v1, v2, v3) == (True, True, True)

    def test_n_papers_zero_no_crash(self):
        # n_papers=0 不应除零崩溃
        v1, v2, v3 = LiteratureTool._triple_verify(
            {"claim": "测试", "paper_idx": []}, n_papers=0
        )
        # n_papers=0 时 coverage=1.0, v3=False
        assert v3 is False


class TestMultiReviewDispatch:
    """multi_review action 分发 + 降级路径."""

    @pytest.mark.asyncio
    async def test_no_papers_returns_error(self):
        """没给 papers 也没给 query → 报错."""
        tool = LiteratureTool()
        args = LiteratureInput(action="multi_review")
        result = await tool.call(args, context=None)
        assert not result.success
        assert "no papers" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_lens_failure_degradation(self):
        """mock LLM: 2 个透镜返回有效 JSON, 1 个返回垃圾.
        失败透镜进 lenses_failed, 成功透镜 findings 正常返回.
        验证 nuwa failure degradation 路径.
        """
        tool = LiteratureTool()
        papers = [
            {"title": "Paper A", "authors": ["X"], "year": 2020, "doi": "10.1/a",
             "abstract": "测试摘要 A"},
            {"title": "Paper B", "authors": ["Y"], "year": 2021, "doi": "10.1/b",
             "abstract": "测试摘要 B"},
            {"title": "Paper C", "authors": ["Z"], "year": 2022, "doi": "10.1/c",
             "abstract": "测试摘要 C"},
        ]
        args = LiteratureInput(
            action="multi_review",
            papers=papers,
            lenses=["methodology", "contributions", "limitations"],
            verify_claims=True,
        )

        # mock _llm_invoke: methodology 返回有效, contributions 返回垃圾, limitations 返回有效
        call_count = 0

        async def mock_llm(model, system_prompt, user_prompt):
            nonlocal call_count
            call_count += 1
            if "methodology" in system_prompt:
                return '{"lens":"methodology","findings":[{"claim":"如果样本量够则结果可信","paper_idx":[1,2],"confidence":"medium"}],"summary":"方法整体OK"}'
            if "contributions" in system_prompt:
                return "这不是JSON垃圾内容"
            if "limitations" in system_prompt:
                return '{"lens":"limitations","findings":[{"claim":"未测试高温场景","paper_idx":[1],"confidence":"high"}],"summary":"有局限"}'
            return ""

        # mock _get_model 避免真初始化 LLM
        with patch.object(LiteratureTool, "_get_model", return_value="fake_model"), \
             patch.object(LiteratureTool, "_llm_invoke", side_effect=mock_llm):
            result = await tool.call(args, context=None)

        assert result.success
        data = result.data
        assert data["action"] == "multi_review"
        assert data["n_lenses_requested"] == 3
        assert data["n_lenses_ok"] == 2  # contributions 挂了
        assert "contributions" in data["lenses_failed"]
        assert len(data["lenses_failed"]) == 1
        # methodology + limitations 各 1 条 claim
        assert data["n_claims_total"] == 2
        # methodology 的 claim: paper_idx=[1,2] n=3 → V1=True, "如果...则" → V2=True, 2/3=0.67 → V3=False
        # score=2 → high
        assert data["n_claims_high"] == 1
        # limitations 的 claim: paper_idx=[1] n=3 → V1=False, 无预测词 → V2=False, 1/3=0.33 < 0.5 → V3=True
        # score=1 → medium
        medium_claims = [c for c in data["all_claims"] if c["final_confidence"] == "medium"]
        assert len(medium_claims) == 1
        assert "未测试高温场景" in medium_claims[0]["claim"]

    @pytest.mark.asyncio
    async def test_verify_claims_disabled(self):
        """verify_claims=False 时跳过三重验证, 直接用 LLM 自报置信度."""
        tool = LiteratureTool()
        papers = [{"title": "P", "authors": [], "year": 2020, "abstract": "x"}]
        args = LiteratureInput(
            action="multi_review",
            papers=papers,
            lenses=["methodology"],
            verify_claims=False,
        )

        async def mock_llm(model, system_prompt, user_prompt):
            return '{"lens":"methodology","findings":[{"claim":"test","paper_idx":[1],"confidence":"high"}],"summary":"ok"}'

        with patch.object(LiteratureTool, "_get_model", return_value="fake"), \
             patch.object(LiteratureTool, "_llm_invoke", side_effect=mock_llm):
            result = await tool.call(args, context=None)

        assert result.success
        assert result.data["verification_enabled"] is False
        # 没有 v1/v2/v3 字段 (跳过验证)
        claim = result.data["all_claims"][0]
        assert "v1_cross_domain" not in claim
        assert claim["final_confidence"] == "high"  # 直接用 LLM 自报
