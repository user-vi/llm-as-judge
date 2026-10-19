# n8n-pochta

Классификация входящих email (корпоративный электронный документооборот).

## Установка

**Python 3.12** 
```bash
uv sync
source .venv/bin/activate
```

## Запуск

### Запуск основного скрипта

```
bash run.sh
```

Итоговый оптимизированный промпт сохраняется в `prompts/<таймстамп>__prompt_<mode>.md`.

### Просмотр лучшего скора

В отдельном терминале (пока `src/main.py` работает или после завернешения) запустите:

```
bash ./best_score.sh
```

Скрипт каждые 5 секунд парсит `gepa.log` и выводит последнюю строку с паттерном `Best score on valset`.

## Структура проекта

```
n8n-pochta/
├── data/
├── src/
│   ├── main.py                # основной entry-point (вызывается из run.sh)
│   ├── finetuning.py          # метрики, PromptOptimizerBase
│   ├── mask_data.py           # маскирование данных
│   ├── prompt.py              # банк стартовых промптов по сценариям
│   ├── logger_setup.py        # настройка логгера
│   └── llm_test.ipynb         # проверка доступности LLM
├── prompts/                   # оптимизированные промпты (сохраняются с таймстампом)
├── run.sh                     # скрипт запуска основного пайплайна src/main.py
├── best_score.sh              # live-watch лучшего скора в gepa.log
├── gepa.log                   # лог GEPA-оптимизации
├── pyproject.toml             # зависимости (uv)
└── README.md
```


## Что делает GEPA

Подробнее о GEPA-оптимизации см. туториалы DSPy:
- [AIME](https://dspy.ai/tutorials/gepa_aime/)
- [Papillon](https://dspy.ai/tutorials/gepa_papillon/)
- [FacilitySupport](https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/)
- [TrustedMonitor](https://dspy.ai/tutorials/gepa_trusted_monitor/)
