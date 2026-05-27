"""Главный orchestrator: corpus + captures → judge → метрики → CSV + summary.

Запуск:
    export DEEPSEEK_API_KEY=sk-...
    python eval/run_eval.py --corpus corpus/scenarios --captures captures --out results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Локальные модули (eval/ в sys.path при запуске из корня)
sys.path.insert(0, str(Path(__file__).parent))

from judge import Judge, JudgeConfig, MockJudge  # noqa: E402
from loaders import load_captures, load_scenarios  # noqa: E402
from metrics import eval_qa, eval_snapshots  # noqa: E402
import report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="memory-bench evaluation")
    ap.add_argument("--corpus", default="corpus/scenarios", type=Path)
    ap.add_argument("--captures", default="captures", type=Path)
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--model", default=None, help="judge model override")
    ap.add_argument("--mock", action="store_true",
                    help="лексический mock-judge без API (для теста пайплайна без денег)")
    args = ap.parse_args()

    scenarios = load_scenarios(args.corpus)
    captures = load_captures(args.captures)
    print(f"[eval] scenarios: {len(scenarios)}, captures: {len(captures)}")

    missing = set(scenarios) - set(captures)
    if missing:
        print(f"[eval] WARN: нет captures для {len(missing)} сценариев: {sorted(missing)}")
    if not captures:
        print("[eval] нет captures — сначала прогони Kotlin bench (adaptive-llm-bench).")
        return 1

    if args.mock:
        judge = MockJudge()
        print(f"[eval] judge: {judge.name} (⚠️ заглушка, не финальные метрики)")
    else:
        cfg = JudgeConfig(model=args.model) if args.model else JudgeConfig()
        try:
            judge = Judge(cfg)
        except RuntimeError as e:
            print(f"[eval] judge init failed: {e}")
            print("[eval] подсказка: для теста пайплайна без API запусти с --mock")
            return 2
        print(f"[eval] judge: {judge.name}")

    qa_rows: list[dict] = []
    snap_rows: list[dict] = []
    for sid, capture in captures.items():
        scenario = scenarios.get(sid)
        if scenario is None:
            print(f"[eval] WARN: capture без сценария в корпусе: {sid}")
            continue
        print(f"[eval] judging {sid} ...")
        qa_rows += eval_qa(sid, capture, judge)
        snap_rows += eval_snapshots(sid, scenario, capture, judge)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_and_summarize(qa_rows, snap_rows, args.out)
    return 0


def _write_and_summarize(qa_rows: list[dict], snap_rows: list[dict], out: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[eval] pandas не установлен — пишу сырой CSV вручную")
        _write_raw_csv(qa_rows, out / "qa.csv")
        _write_raw_csv(snap_rows, out / "snapshots.csv")
        return

    qa_df = pd.DataFrame(qa_rows)
    snap_df = pd.DataFrame(snap_rows)
    qa_df.to_csv(out / "qa.csv", index=False)
    snap_df.to_csv(out / "snapshots.csv", index=False)

    print("\n=== QA accuracy by type ===")
    if not qa_df.empty:
        by_type = qa_df.groupby("qa_type")["passed"].agg(["mean", "count"])
        print(by_type.to_string())
        print(f"\noverall QA pass rate: {qa_df['passed'].mean():.1%} (n={len(qa_df)})")

    print("\n=== DB diagnostics (mean) ===")
    if not snap_df.empty:
        print(f"extraction recall: {snap_df['extraction_recall'].mean():.1%}")
        print(f"summary fidelity:  {snap_df['summary_fidelity'].mean():.1%}")

    print(f"\n[eval] CSV → {out}/qa.csv, {out}/snapshots.csv")

    # Графики + summary.md
    report.write_report(qa_df, snap_df, out)


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
