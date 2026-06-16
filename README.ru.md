[English](README.md) | **Русский**

# memory-bench

Бенчмарк системы долговременной памяти для on-device LLM ассистента. Корпус диалоговых сценариев (20 авторских RU + 40 LongMemEval-ru = 60) + Python оценки.

## Архитектура

Двухкомпонентная, прогон системы и судейство — разные задачи:

```
adaptive-llm-bench/  (Kotlin, отдельный репо)
  прогоняет сценарий через настоящую систему памяти (llama.cpp + sqlite-vec +
  USER2-small embedding через JNI), захватывает:
    - QA-ответы модели
    - DB snapshots (факты + summary на checkpoint'ах)
    - retrieved top-K для каждого QA
  → пишет captures/<scenario>-capture.json

memory-bench/  (этот репо)
  corpus/scenarios/  — 20 авторских RU-сценариев (ru-*, gold/expected)
  corpus/SCHEMA.md   — формат сценариев
  longmemeval-ru/    — 40 сценариев LongMemEval, переведённых на русский (lme-*)
  eval/    — Python: читает scenario gold + capture.json → DeepSeek judge →
             метрики → CSV + графики
  captures/, captures-lme/ — сюда Kotlin bench кладёт сырые прогоны (gitignored)
  results/, results-lme/   — метрики и графики (gitignored)
```

Почему так: корпус — переиспользуемая спецификация (любая memory-система может
прогнаться). Judge на Python — удобнее API-клиенты, pandas, matplotlib для
таблиц/графиков диплома. Kotlin делает «дорогую» часть (модель через JNI).

## Метрики (см. SCHEMA.md для деталей сценариев)

**Primary — end-to-end QA** (LoCoMo/LongMemEval-style):
- `recall` — одиночный факт извлечён и применён в ответе
- `multi_fact` — несколько фактов скомбинированы в ответе
- `abstention` — модель не выдумывает то, чего не было
- `update` — отвечает актуальным значением, не stale
- `temporal` — корректно понимает абсолютные даты
- `gist` — модель кратко передаёт суть разговора («чем мы занимались?») — тестирует summary, а не точечные факты
- `synthesis` — связный итог истории, требует склейки фактов и summary

**Secondary — диагностика:**
- retrieval top-K hit (всплыл ли нужный факт)
- DB snapshot (extraction recall, conflict resolution, summary fidelity)

## Запуск

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...        # judge

# авторский RU-корпус (20 сценариев)
python eval/run_eval.py --corpus corpus/scenarios --captures captures --out results

# LongMemEval-ru (40 сценариев) — отдельный прогон
python eval/run_eval.py --corpus longmemeval-ru --captures captures-lme --out results-lme
```

## Judge

DeepSeek V4-Pro (open weights, дёшево, воспроизводимо) как primary judge
