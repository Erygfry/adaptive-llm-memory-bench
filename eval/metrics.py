"""Вычисление метрик: primary QA + secondary DB-диагностика.

Возвращает «длинные» строки (по одной на проверку) — удобно для pandas
агрегации в report.py.
"""
from __future__ import annotations

from judge import Judge


def _mention_hits(answer: str, needles: list[str]) -> list[str]:
    low = answer.lower()
    return [n for n in needles if n.lower() in low]


def eval_qa(scenario_id: str, capture: dict, judge: Judge) -> list[dict]:
    """Оценка QA-ответов (primary). Одна строка на QA-probe."""
    rows: list[dict] = []
    for qa in capture.get("qaCaptures", []):
        # QaCheck сериализуется kotlinx с @SerialName в snake_case
        check = qa["check"]
        qa_type = check["type"]
        gold = check["gold"]
        must = check.get("must_mention", [])
        must_not = check.get("must_not_mention", [])
        instr = check.get("instruction_compliance")
        answer = qa["answer"]

        judged = judge.judge_qa(qa_type, qa["question"], gold, answer)

        must_hits = _mention_hits(answer, must)
        must_pass = (len(must_hits) == len(must)) if must else True
        must_not_viol = _mention_hits(answer, must_not)
        must_not_pass = len(must_not_viol) == 0

        instr_pass = None
        if instr:
            instr_pass = judge.judge_instruction_compliance(instr, answer)

        # retrieval диагностика: всплыло ли что-то в принципе
        retrieved_n = len(qa.get("retrievedTopK", []))

        rows.append({
            "scenario": scenario_id,
            "kind": "qa",
            "turn": qa["turn"],
            "qa_type": qa_type,
            "judge_correct": judged,
            "must_mention_pass": must_pass,
            "must_not_mention_pass": must_not_pass,
            "instruction_pass": instr_pass,
            "retrieved_count": retrieved_n,
            # passed = judge + детерминистичные гейты
            "passed": judged and must_pass and must_not_pass and (instr_pass is not False),
        })
    return rows


def eval_snapshots(scenario_id: str, scenario: dict, capture: dict, judge: Judge) -> list[dict]:
    """DB-диагностика (secondary): extraction recall + summary fidelity."""
    rows: list[dict] = []
    checkpoints = {c["after_turn"]: c for c in scenario.get("db_checkpoints", [])}
    for snap in capture.get("snapshots", []):
        cp = checkpoints.get(snap["afterTurn"])
        if cp is None:
            continue
        active = snap.get("activeFacts", [])

        # extraction recall: каждый expected факт ищем среди captured
        expected = cp.get("expected_active_facts", [])
        matched = 0
        for exp in expected:
            paraphrases = exp["content_paraphrases"]
            if any(judge.fact_equivalent(f["content"], paraphrases) for f in active):
                matched += 1
        recall = matched / len(expected) if expected else 1.0

        # summary fidelity
        summary_text = " ".join(filter(None, [
            (snap.get("summary") or {}).get("userProfile", ""),
            (snap.get("summary") or {}).get("ongoingTopics", ""),
            (snap.get("summary") or {}).get("keyDecisions", ""),
            (snap.get("summary") or {}).get("pendingItems", ""),
        ]))
        concepts = cp.get("expected_summary_includes", [])
        present = sum(1 for c in concepts if judge.summary_contains(summary_text, c))
        fidelity = present / len(concepts) if concepts else 1.0

        rows.append({
            "scenario": scenario_id,
            "kind": "snapshot",
            "after_turn": snap["afterTurn"],
            "extraction_recall": recall,
            "extraction_matched": matched,
            "extraction_expected": len(expected),
            "summary_fidelity": fidelity,
            "active_facts": len(active),
        })
    return rows
