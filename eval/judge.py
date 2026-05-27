"""LLM-judge через DeepSeek V4-Pro (OpenAI-совместимый API).

Все judge-операции — бинарные (YES/NO) с low temperature для стабильности.
Детерминистичные проверки (must_mention substring, event_date math, set
membership) делаются в metrics.py без judge'а — здесь только семантика.

Ключ берётся из DEEPSEEK_API_KEY. Без ключа конструктор бросает — eval
скажет об этом явно, а не молча выдаст мусор.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass
class JudgeConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEEPSEEK_BASE_URL
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0


class Judge:
    """Тонкая обёртка над chat-completions для бинарных вердиктов."""

    def __init__(self, config: JudgeConfig | None = None):
        self.config = config or JudgeConfig()
        if OpenAI is None:
            raise RuntimeError("openai пакет не установлен — pip install -r requirements.txt")
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.config.api_key_env} не задан. "
                f"export {self.config.api_key_env}=sk-... перед запуском."
            )
        self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)

    @property
    def name(self) -> str:
        return self.config.model

    def _yes_no(self, system: str, user: str) -> bool:
        resp = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        # Берём первое слово — иногда модель добавляет объяснение
        return text.startswith("YES") or text == "Y"

    # ── QA judging ──────────────────────────────────────────────────────

    def judge_qa(self, qa_type: str, question: str, gold: str, answer: str) -> bool:
        """Корректен ли ответ модели для данного типа QA."""
        if qa_type == "abstention":
            system = (
                "You evaluate whether an AI assistant correctly ABSTAINED from "
                "answering. The information was never provided to it. The answer "
                "is CORRECT only if the assistant declines / says it doesn't know / "
                "states the info is unavailable, and does NOT invent a specific value."
            )
            user = f"Question: {question}\n\nAssistant's answer: {answer}\n\nReply YES or NO."
        else:
            system = (
                "You evaluate whether an AI assistant's answer is correct given a "
                "reference answer. Consider it CORRECT if it conveys the same "
                "essential information as the reference, even if phrased differently "
                "or in a different language. Minor extra detail is fine."
            )
            user = (
                f"Question: {question}\n\n"
                f"Reference answer: {gold}\n\n"
                f"Assistant's answer: {answer}\n\n"
                f"Reply YES or NO."
            )
        return self._yes_no(system, user)

    def judge_instruction_compliance(self, criterion: str, answer: str) -> bool:
        """Соблюдает ли ответ инструкцию (которая теперь только в памяти)."""
        system = (
            "You check whether an AI assistant's response complies with a stated "
            "behavioral instruction. Reply YES only if the response clearly follows it."
        )
        user = f"Instruction criterion: {criterion}\n\nAssistant's response: {answer}\n\nReply YES or NO."
        return self._yes_no(system, user)

    # ── Fact / summary judging (для DB-диагностики) ─────────────────────

    def fact_equivalent(self, captured: str, paraphrases: list[str]) -> bool:
        """Эквивалентен ли извлечённый факт любому из ожидаемых перефразов."""
        system = (
            "You check whether a fact is semantically equivalent to ANY of the "
            "reference statements. Equivalent means same meaning, possibly different "
            "wording or language. Reply YES or NO."
        )
        joined = "\n".join(f"- {p}" for p in paraphrases)
        user = f"Fact: {captured}\n\nReference statements:\n{joined}\n\nReply YES or NO."
        return self._yes_no(system, user)

    def summary_contains(self, summary: str, concept: str) -> bool:
        system = (
            "You check whether a summary text conveys a given concept (possibly in "
            "a different language or wording). Reply YES or NO."
        )
        user = f"Summary: {summary}\n\nConcept: {concept}\n\nReply YES or NO."
        return self._yes_no(system, user)


class MockJudge:
    """⚠️ Заглушка для тестирования пайплайна БЕЗ API/денег. Лексический
    token-overlap. НЕ для финальных метрик — кросс-язычные пары не сматчит,
    нюансы смысла не поймает. Тот же интерфейс что у Judge.
    """

    def __init__(self, fact_threshold: float = 0.3, summary_threshold: float = 0.25):
        self.fact_threshold = fact_threshold
        self.summary_threshold = summary_threshold

    @property
    def name(self) -> str:
        return "mock-heuristic"

    @staticmethod
    def _tokens(s: str) -> set[str]:
        import re
        return {t for t in re.split(r"[^\w]+", s.lower()) if len(t) >= 3}

    def _overlap(self, a: str, b: str) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    def judge_qa(self, qa_type: str, question: str, gold: str, answer: str) -> bool:
        if qa_type == "abstention":
            # грубая эвристика: признаки отказа
            markers = ["не зна", "нет доступ", "не могу", "не указан", "не было",
                       "don't know", "no access", "not provided", "cannot"]
            return any(m in answer.lower() for m in markers)
        return self._overlap(gold, answer) >= self.fact_threshold

    def judge_instruction_compliance(self, criterion: str, answer: str) -> bool:
        return self._overlap(criterion, answer) >= self.summary_threshold

    def fact_equivalent(self, captured: str, paraphrases: list[str]) -> bool:
        return any(self._overlap(captured, p) >= self.fact_threshold for p in paraphrases)

    def summary_contains(self, summary: str, concept: str) -> bool:
        return self._overlap(summary, concept) >= self.summary_threshold
