# n8n-pochta

Классификация входящих email (корпоративный электронный документооборот).

## Установка

**Python 3.12** 
```bash
uv sync
source .venv/bin/activate
```

## Запуск

### Проверка доступности LLM

Перед запуском основного скрипта проверить, что LLM-модель доступна. Запустить [`src/llm_test.ipynb`](src/llm_test.ipynb). 
 * Модель доступна только из dev-контура
 * Модель могут заменить


### Запуск основного скрипта

```
python src/main.py --mode <нужный_режим>
```

#### Возможные значения `--mode`

Флаг `--mode` обязательный — задаёт сценарий классификации. Поддерживаются следующие значения:

| Режим | Фильтр | Датасет | Метрика |
|---|---|---|---|
| `recipient_kos` | `kos@kos.sibur.ru` | `data/eval_data__recipient.xlsx` | `judge_metric_float` |
| `recipient_nknh` | `nknh@sibur.ru` | `data/eval_data__recipient.xlsx` | `judge_metric_float` |
| `recipient_mos` | `info@sibur.ru` | `data/eval_data__recipient.xlsx` | `judge_metric_float` |
| `forward_kos` | `kos@kos.sibur.ru` | `data/eval_data__forward.xlsx` | `judge_metric_bin` |
| `forward_nknh` | `nknh@sibur.ru` | `data/eval_data__forward.xlsx` | `judge_metric_bin` |
| `forward_mos` | `info@sibur.ru` | `data/eval_data__forward.xlsx` | `judge_metric_bin` |
| `kis_kos` | `kos@kos.sibur.ru` | `data/eval_data__is_kisecm.xlsx` | `judge_metric_bin` |
| `kis_nknh` | `nknh@sibur.ru` | `data/eval_data__is_kisecm.xlsx` | `judge_metric_bin` |
| `kis_mos` | `info@sibur.ru` | `data/eval_data__is_kisecm.xlsx` | `judge_metric_bin` |
| `isGD` | без фильтра | `data/eval_data__isGD.xlsx` | `judge_metric_bin` |
| `deadline` | без фильтра | `data/eval_data__deadline.xlsx` | `judge_metric_bin` |
| `sender` | без фильтра | `data/eval_data__sender.xlsx` | `judge_metric_bin` |


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
│   ├── eval_data__recipient.xlsx      # разметка для recipient_kos, recipient_nknh, recipient_mos
│   ├── eval_data__forward.xlsx        # разметка для forward_kos, forward_nknh, forward_mos
│   ├── eval_data__is_kisecm.xlsx      # разметка для kis_kos, kis_nknh, kis_mos
│   ├── eval_data__isGD.xlsx           # разметка для isGD
│   ├── eval_data__deadline.xlsx       # разметка для deadline
│   └── eval_data__sender.xlsx         # разметка для sender
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
