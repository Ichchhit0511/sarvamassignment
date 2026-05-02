"""Run the golden eval set against the live pipeline and print accuracy metrics.

Three accuracy axes are measured:

  1. Retrieval Recall@5
       Did at least one of the expected pages appear in the top-5 retrieved chunks?
       Measures: is the right info reaching the LLM?

  2. Refusal Accuracy
       For out-of-manual questions: did the system correctly refuse
       (manual_supported=False)?
       Measures: is the grounding contract holding?

  3. Faithfulness (Gemini-as-judge)
       For in-manual questions: is the generated answer entailed by the
       retrieved chunks? Scored 1-5 by Gemini, ≥4 counts as faithful.
       Measures: is Sarvam staying grounded?

Usage:
    python -m scripts.evaluate                       # uses data/eval/golden.jsonl
    python -m scripts.evaluate path/to/other.jsonl

Each row in the JSONL must have:
    id, manual_id, query, language,
    expected_pages: [int, ...]    (empty if should_refuse)
    expected_keywords: [...]      (English)
    expected_keywords_translit: [...]  (transliteration to look for in non-EN answers)
    should_refuse: bool
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.generator import generate
from backend.query_rewriter import rewrite
from backend.retrieval import retrieve
from backend.verifier import verify
from backend.vision import describe_image


# ---------------------------------------------------------------------------
# Faithfulness via Gemini-as-judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are evaluating whether an ANSWER is faithful to the SOURCE.
Faithful = every claim in ANSWER is directly supported by SOURCE. Translation
into another language is allowed; adding new facts is NOT.

SOURCE (English manual chunks):
{source}

ANSWER (may be in any language):
{answer}

Rate faithfulness on this 1-5 scale:
  5 = every claim entailed by source; no additions
  4 = mostly entailed, minor wording embellishments
  3 = partial — some claims supported, some not
  2 = mostly unsupported
  1 = wholly fabricated

Output STRICT JSON: {{"score": <int 1-5>, "reason": "<one sentence>"}}
"""


def _gemini_judge(answer_text: str, chunks_text: str) -> tuple[int, str]:
    if not settings.gemini_api_key:
        return 3, "(judge skipped — no GEMINI_API_KEY)"
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_rewriter_model)
    prompt = JUDGE_PROMPT.format(source=chunks_text[:6000], answer=answer_text[:3000])
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            return int(obj.get("score", 3)), str(obj.get("reason", ""))
    except Exception as e:
        return 3, f"(judge error: {e})"
    return 3, "(judge unparsable)"


# ---------------------------------------------------------------------------
# Single-row eval
# ---------------------------------------------------------------------------
def evaluate_row(row: dict) -> dict:
    t0 = time.time()
    rewrites, _ = rewrite(row["query"], vision_text="")
    retrieved = retrieve(
        queries=rewrites,
        original_query=row["query"],
        top_k=5,
        manual_id=row["manual_id"],
    )
    raw, _ = generate(row["query"], None, retrieved)
    answer = verify(raw, retrieved)
    elapsed = time.time() - t0

    retrieved_pages = sorted({rc.chunk.page for rc in retrieved})
    expected_pages = set(row.get("expected_pages") or [])
    recall_hit = (
        bool(expected_pages & set(retrieved_pages))
        if expected_pages else None
    )

    should_refuse = bool(row.get("should_refuse"))
    refused = not answer.manual_supported
    refusal_correct = (refused == should_refuse)

    # Faithfulness only for non-refusal cases.
    faithfulness_score: int | None = None
    judge_reason = ""
    if not should_refuse and answer.manual_supported:
        chunks_text = "\n\n".join(
            f"[p.{rc.chunk.page}] {rc.chunk.text}" for rc in retrieved
        )
        faithfulness_score, judge_reason = _gemini_judge(answer.answer, chunks_text)

    # Keyword presence in the answer (rough sanity check).
    kw_field = (
        "expected_keywords" if row.get("language") == "en" else "expected_keywords_translit"
    )
    expected_kws = row.get(kw_field) or []
    answer_low = answer.answer.lower()
    kw_hits = [kw for kw in expected_kws if kw.lower() in answer_low]
    kw_recall = (len(kw_hits) / len(expected_kws)) if expected_kws else None

    return {
        "id": row["id"],
        "language": row.get("language"),
        "should_refuse": should_refuse,
        "refused": refused,
        "refusal_correct": refusal_correct,
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": retrieved_pages,
        "recall_hit": recall_hit,
        "kw_recall": kw_recall,
        "kw_hits": kw_hits,
        "faithfulness": faithfulness_score,
        "judge_reason": judge_reason,
        "manual_supported": answer.manual_supported,
        "confidence": answer.confidence,
        "answer_preview": answer.answer[:200],
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Aggregate + pretty print
# ---------------------------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    in_manual = [r for r in rows if not r["should_refuse"]]
    out_manual = [r for r in rows if r["should_refuse"]]

    recall_hits = [r for r in in_manual if r["recall_hit"]]
    refusal_correct = [r for r in rows if r["refusal_correct"]]
    faithful = [r for r in in_manual if (r["faithfulness"] or 0) >= 4]
    kw_scored = [r for r in in_manual if r["kw_recall"] is not None]
    avg_kw = (sum(r["kw_recall"] for r in kw_scored) / len(kw_scored)) if kw_scored else 0.0
    avg_latency = sum(r["elapsed_s"] for r in rows) / n if n else 0.0

    return {
        "n_total": n,
        "n_in_manual": len(in_manual),
        "n_out_manual": len(out_manual),
        "retrieval_recall_at_5": (len(recall_hits) / len(in_manual)) if in_manual else 0.0,
        "refusal_accuracy": (len(refusal_correct) / n) if n else 0.0,
        "faithfulness_at_4plus": (len(faithful) / len(in_manual)) if in_manual else 0.0,
        "avg_keyword_recall": avg_kw,
        "avg_latency_s": round(avg_latency, 2),
    }


def main() -> None:
    eval_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data/eval/golden.jsonl"
    if not eval_path.exists():
        sys.exit(f"❌ Eval set not found: {eval_path}")

    rows_in = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    print(f"Running {len(rows_in)} eval rows from {eval_path.name}...\n")

    results: list[dict] = []
    for i, row in enumerate(rows_in, 1):
        print(f"[{i:>2}/{len(rows_in)}] {row['id']}: {row['query'][:60]}")
        try:
            r = evaluate_row(row)
        except Exception as e:
            r = {"id": row["id"], "error": str(e), "should_refuse": False,
                 "refused": False, "refusal_correct": False, "recall_hit": None,
                 "kw_recall": None, "faithfulness": None, "elapsed_s": 0}
        results.append(r)
        marks = []
        if r.get("recall_hit") is True: marks.append("✅recall")
        elif r.get("recall_hit") is False: marks.append("❌recall")
        if r.get("refusal_correct"): marks.append("✅refusal")
        else: marks.append("❌refusal")
        if r.get("faithfulness"): marks.append(f"⭐{r['faithfulness']}")
        print(f"      {' '.join(marks)}  ({r.get('elapsed_s', 0):.1f}s)")

    summary = summarize(results)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total queries:               {summary['n_total']}")
    print(f"    in-manual:                 {summary['n_in_manual']}")
    print(f"    out-of-manual:             {summary['n_out_manual']}")
    print(f"  Retrieval Recall@5:          {summary['retrieval_recall_at_5']*100:5.1f}%")
    print(f"  Refusal Accuracy:            {summary['refusal_accuracy']*100:5.1f}%")
    print(f"  Faithfulness ≥4 (LLM-judge): {summary['faithfulness_at_4plus']*100:5.1f}%")
    print(f"  Avg keyword recall:          {summary['avg_keyword_recall']*100:5.1f}%")
    print(f"  Avg latency:                 {summary['avg_latency_s']:.2f} s")

    out_path = PROJECT_ROOT / "data/eval/last_run.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    print(f"\n  Detail written to: {out_path}")


if __name__ == "__main__":
    main()
