"""Citation verification — guardrail layer.

After Sarvam produces the grounded answer, we cross-check that the cited
chunks actually exist in the retrieved set. This catches the most common
hallucination mode (LLM invents a page number).

For deeper checking, you can additionally embed each sentence and compare
against the cited chunk; we keep that lightweight here to stay snappy.
"""
from __future__ import annotations

from .models import GroundedAnswer, RetrievedChunk


def verify(answer: GroundedAnswer,
           retrieved: list[RetrievedChunk]) -> GroundedAnswer:
    if not answer.manual_supported:
        return answer

    valid_ids = {rc.chunk.chunk_id for rc in retrieved}
    valid_pages = {rc.chunk.page for rc in retrieved}

    kept = [
        c for c in answer.citations
        if c.chunk_id in valid_ids or c.page in valid_pages
    ]

    if not kept:
        # Sarvam claimed manual support but cited nothing real — downgrade.
        return GroundedAnswer(
            answer=answer.answer,
            citations=[],
            confidence="low",
            manual_supported=False,
            language=answer.language,
        )

    return GroundedAnswer(
        answer=answer.answer,
        citations=kept,
        confidence=answer.confidence,
        manual_supported=True,
        language=answer.language,
    )
