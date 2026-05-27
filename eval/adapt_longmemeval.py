"""Конвертер LongMemEval (oracle) → наша scenario-схема.

LongMemEval уже в формате user↔assistant (haystack_sessions = списки
{role, content}), что совпадает с нашей парадигмой. Конвертим выборку
вопросов в сценарии: сессии → dialog (с датами), вопрос → финальный QA-probe.

Выбираем oracle-версию (только answer-содержащие сессии, без дистракторов) —
чистая основа. _s/_m версии с дистракторами слишком большие и тестируют
retrieval-at-scale, что у нас отдельная задача.

Запуск:
    python eval/adapt_longmemeval.py --n-per-type 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# LongMemEval question_type → наш qa.type
TYPE_MAP = {
    "temporal-reasoning": "temporal",
    "multi-session": "multi_fact",
    "knowledge-update": "update",
    "single-session-user": "recall",
    "single-session-preference": "recall",
    # single-session-assistant пропускаем: спрашивает про реплики ассистента,
    # а наша система хранит факты о ПОЛЬЗОВАТЕЛЕ, не о себе.
}

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def parse_date(s: str) -> str:
    """'2023/05/25 (Thu) 20:21' → '2023-05-25'."""
    m = _DATE_RE.search(s)
    if not m:
        return "2023-01-01"
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def build_scenario(item: dict) -> dict | None:
    qid = item["question_id"]
    is_abstention = qid.endswith("_abs")
    lme_type = item["question_type"]

    if is_abstention:
        qa_type = "abstention"
    else:
        qa_type = TYPE_MAP.get(lme_type)
        if qa_type is None:
            return None

    sessions = item["haystack_sessions"]
    dates = [parse_date(d) for d in item["haystack_dates"]]

    dialog: list[dict] = []
    force_after: list[int] = []
    turn = 0
    for si, session in enumerate(sessions):
        sess_date = dates[si] if si < len(dates) else dates[-1]
        for ti, msg in enumerate(session):
            turn += 1
            entry = {"turn": turn, "role": msg["role"], "content": msg["content"]}
            if ti == 0:
                entry["date"] = sess_date
            dialog.append(entry)
        force_after.append(turn)  # eviction после каждой сессии

    # Финальный QA-probe — сам вопрос
    turn += 1
    gold = ("The assistant should decline / say it does not know — the information "
            "was never provided." if is_abstention else str(item["answer"]))
    dialog.append({
        "turn": turn,
        "role": "user",
        "content": item["question"],
        "date": parse_date(item["question_date"]),
        "checks": {"qa": {"type": qa_type, "gold": gold}},
    })

    short = qid.replace("gpt4_", "")[:12]
    return {
        "id": f"lme-{qa_type}-{short}",
        "description": f"Adapted from LongMemEval ({lme_type}, id={qid}).",
        "category": "mixed",
        "language": "en",
        "starting_date": dates[0] if dates else "2023-01-01",
        "ram_tier": "MID",
        "eviction_policy": {"mode": "force", "force_after_turns": force_after},
        "dialog": dialog,
        "db_checkpoints": [],  # LongMemEval не даёт gold-фактов, только QA
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="reference/longmemeval_oracle.json", type=Path)
    ap.add_argument("--out", default="corpus/scenarios", type=Path)
    ap.add_argument("--n-per-type", default=3, type=int,
                    help="сколько сценариев на каждый qa-тип")
    ap.add_argument("--max-turns", default=80, type=int,
                    help="пропускать вопросы где диалог длиннее (слишком долгий прогон)")
    args = ap.parse_args()

    data = json.loads(args.reference.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    written = 0
    for item in data:
        sc = build_scenario(item)
        if sc is None:
            continue
        qa_type = sc["dialog"][-1]["checks"]["qa"]["type"]
        if counts.get(qa_type, 0) >= args.n_per_type:
            continue
        if len(sc["dialog"]) > args.max_turns:
            continue
        (args.out / f"{sc['id']}.json").write_text(
            json.dumps(sc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts[qa_type] = counts.get(qa_type, 0) + 1
        written += 1

    print(f"[adapt] wrote {written} scenarios: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
