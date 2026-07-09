"""Tests for value-aware context retention in summarize_compact_messages."""

import asyncio

import pytest


class TestMessageValueScore:
    """Verify _message_value_score correctly ranks messages."""

    def test_energy_keyword_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("total energy = -10.5 eV") > 0

    def test_bandgap_keyword_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("band gap = 1.2 eV, direct") > 0

    def test_convergence_keyword_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("SCF converged after 12 iterations") > 0

    def test_lattice_keyword_positive(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("lattice: a = 5.43 b = 5.43 c = 5.43") > 0

    def test_debug_negative(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("debug: loading model checkpoint") < 0

    def test_traceback_negative(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("traceback (most recent call last)") < 0

    def test_plain_text_zero(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("hello world, how are you") == 0

    def test_empty_string_zero(self):
        from huginn.utils.context import _message_value_score

        assert _message_value_score("") == 0

    def test_dict_with_energy_positive(self):
        from huginn.utils.context import _message_value_score

        msg = {"role": "assistant", "content": "energy = -10.5 eV"}
        assert _message_value_score(msg) > 0

    def test_dict_debug_negative(self):
        from huginn.utils.context import _message_value_score

        msg = {"role": "user", "content": "debug: step 1 completed"}
        assert _message_value_score(msg) < 0

    def test_tool_message_with_data_bonus(self):
        """Long tool output gets +1 even without keyword match."""
        from huginn.utils.context import _message_value_score

        msg = {"role": "tool", "content": "x" * 200}
        assert _message_value_score(msg) > 0


class TestSummarizeCompactPreservesHighValue:
    """Verify high-value messages survive compaction."""

    def test_high_value_not_sent_to_summarizer(self):
        """High-value messages should be pulled out of summarize_zone,
        so they never appear in the summarizer's transcript."""
        from huginn.utils.context import summarize_compact_messages

        # Pad low-value messages so total tokens exceed budget
        pad = "x" * 500
        messages = [
            {"role": "user", "content": f"debug: loading model. {pad}"},
            {"role": "assistant", "content": "total energy = -10.5 eV, converged=True, encut=520"},
            {"role": "user", "content": f"debug: step 2 done. {pad}"},
            {"role": "assistant", "content": "what is the bandgap?"},
            {"role": "user", "content": "the band gap is 1.2 eV"},
        ]

        captured = []

        async def fake_summarizer(transcript):
            captured.append(transcript)
            return "summary of conversation"

        compacted, _ = asyncio.run(
            summarize_compact_messages(
                messages,
                budget_tokens=100,
                keep_last_n=2,
                summarizer=fake_summarizer,
            )
        )

        # The high-value message should NOT have been sent to the summarizer
        assert captured, "summarizer should have been called"
        assert "-10.5" not in captured[0]
        assert "encut" not in captured[0]

    def test_high_value_survives_in_output(self):
        """High-value message content should appear in the compacted output."""
        from huginn.utils.context import summarize_compact_messages

        pad = "x" * 500
        messages = [
            {"role": "user", "content": f"debug: loading model. {pad}"},
            {"role": "assistant", "content": "total energy = -10.5 eV, converged=True"},
            {"role": "user", "content": f"debug: step 2 done. {pad}"},
            {"role": "assistant", "content": "what is the bandgap?"},
            {"role": "user", "content": "the band gap is 1.2 eV"},
        ]

        async def fake_summarizer(transcript):
            return "summary of conversation"

        compacted, _ = asyncio.run(
            summarize_compact_messages(
                messages,
                budget_tokens=100,
                keep_last_n=2,
                summarizer=fake_summarizer,
            )
        )

        # Extract all text from the compacted messages
        all_text = " ".join(
            m.get("content", "") if isinstance(m, dict)
            else str(getattr(m, "content", ""))
            for m in compacted
        )
        assert "-10.5" in all_text, "high-value energy data should survive compaction"

    def test_low_value_sent_to_summarizer(self):
        """Low-value messages should still be sent to the summarizer."""
        from huginn.utils.context import summarize_compact_messages

        pad = "x" * 500
        messages = [
            {"role": "user", "content": f"debug: loading model. {pad}"},
            {"role": "assistant", "content": "debug: step 2 done. more debug output"},
            {"role": "user", "content": "what is the bandgap?"},
            {"role": "assistant", "content": "the band gap is 1.2 eV"},
        ]

        captured = []

        async def fake_summarizer(transcript):
            captured.append(transcript)
            return "summary of conversation"

        asyncio.run(
            summarize_compact_messages(
                messages,
                budget_tokens=50,
                keep_last_n=2,
                summarizer=fake_summarizer,
            )
        )

        assert captured, "summarizer should have been called"
        # At least one of the debug messages should be in the transcript
        assert "debug" in captured[0]


class TestNoSummarizerFallback:
    """Verify the no-summarizer fallback path still works."""

    def test_fallback_returns_compacted_list(self):
        """Without a summarizer, should fall back to drop-oldest without crashing."""
        from huginn.utils.context import summarize_compact_messages

        pad = "x" * 500
        messages = [
            {"role": "user", "content": f"energy = -10.5 eV {pad}"},
            {"role": "assistant", "content": "great"},
            {"role": "user", "content": f"bandgap is 1.2 eV {pad}"},
            {"role": "assistant", "content": "noted"},
        ]

        compacted, summary = asyncio.run(
            summarize_compact_messages(
                messages,
                budget_tokens=100,
                keep_last_n=2,
                summarizer=None,
            )
        )

        assert isinstance(compacted, list)
        assert len(compacted) <= len(messages)
        # Summary should be unchanged (no summarizer ran)
        assert summary == ""

    def test_fallback_under_budget_returns_unchanged(self):
        """If already under budget, should return messages as-is."""
        from huginn.utils.context import summarize_compact_messages

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        compacted, summary = asyncio.run(
            summarize_compact_messages(
                messages,
                budget_tokens=10000,
                keep_last_n=2,
                summarizer=None,
            )
        )

        assert len(compacted) == len(messages)
