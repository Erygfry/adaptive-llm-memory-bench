**English** | [Русский](README.ru.md)

# memory-bench

A benchmark for the long-term memory system of an on-device LLM assistant. A corpus of dialogue scenarios (20 authored RU + 40 LongMemEval-ru = 60) + Python evaluation.

## Architecture

Two components — running the system and judging are separate concerns:

```
adaptive-llm-bench/  (Kotlin, separate repo)
  runs a scenario through the real memory system (llama.cpp + sqlite-vec +
  USER2-small embedding via JNI), capturing:
    - the model's QA answers
    - DB snapshots (facts + summary at checkpoints)
    - retrieved top-K for each QA
  → writes captures/<scenario>-capture.json

memory-bench/  (this repo)
  corpus/scenarios/  — 20 authored RU scenarios (ru-*, gold/expected)
  corpus/SCHEMA.md   — scenario format
  longmemeval-ru/    — 40 LongMemEval scenarios translated into Russian (lme-*)
  eval/    — Python: reads scenario gold + capture.json → DeepSeek judge →
             metrics → CSV + charts
  captures/, captures-lme/ — the Kotlin bench drops raw runs here (gitignored)
  results/, results-lme/   — metrics and charts (gitignored)
```

Why this split: the corpus is a reusable specification (any memory system can be
run against it). Keeping the judge in Python is more convenient — API clients,
pandas, matplotlib for the thesis tables/charts. Kotlin does the "expensive"
part (the model via JNI).

## Metrics (see corpus/SCHEMA.md for scenario details)

**Primary — end-to-end QA** (LoCoMo/LongMemEval-style):
- `recall` — a single fact is retrieved and used in the answer
- `multi_fact` — several facts are combined in the answer
- `abstention` — the model does not make up things that were never said
- `update` — answers with the current value, not a stale one
- `temporal` — correctly understands absolute dates
- `gist` — the model briefly conveys the gist of the conversation ("what were we doing?") — tests the summary, not point facts
- `synthesis` — a coherent summary of the history, requiring facts and summary to be stitched together

**Secondary — diagnostics:**
- retrieval top-K hit (did the needed fact surface)
- DB snapshot (extraction recall, conflict resolution, summary fidelity)

## Running

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...        # judge

# authored RU corpus (20 scenarios)
python eval/run_eval.py --corpus corpus/scenarios --captures captures --out results

# LongMemEval-ru (40 scenarios) — separate run
python eval/run_eval.py --corpus longmemeval-ru --captures captures-lme --out results-lme
```

## Judge

DeepSeek V4-Pro (open weights, cheap, reproducible) as the primary judge
