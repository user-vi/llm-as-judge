import dspy
import json
import codecs
import litellm
from typing import Any
from dotenv import load_dotenv
from src.logger_setup import setup_dspy_logger

log = setup_dspy_logger(log_file="gepa.log")
log.info("GEPA compile started")

load_dotenv()

litellm.ssl_verify = False  # проверка сертификатов выключена, для ллм

judge_lm = dspy.LM(
    model="openai//qwen36-27b",
    api_key="empty",
    api_base="https://qwen.k8s.dev002.local/v1",
    temperature=0.0,
)

dspy.configure(lm=judge_lm)
dspy.settings.configure(debug=True)


class OptimizationSignature(dspy.Signature):
    """Докстринга меняется через инъекции, является первичным промтом"""

    email = dspy.InputField(desc="Предприятие + Тема + тело письма")
    # reason = dspy.OutputField(desc="Краткое объяснение решения (до 10 слов)")
    # predict: Literal["true", "false"] = dspy.OutputField(desc="Предсказание модели")
    predict = dspy.OutputField(desc="Предсказание модели")


class PromptOptimizerBase(dspy.Module):
    # student = PromptOptimizerRecipientBase(new_doc=receiver_nknh)
    def __init__(self, docstring):
        super().__init__()
        signature = OptimizationSignature
        signature.__doc__ = docstring
        self.generator = dspy.ChainOfThought(signature)

    def forward(self, email):
        pred = self.generator(email=email)
        return pred


class JudgeSignature(dspy.Signature):
    """Ты — ассесор-разметчик бинарной классификации.

    Работай как в TDD (Test-Driven Development). Твоя проверка — это
    оператор assert:

        assert expected == actual   → PASS → 1
        assert expected != actual   → FAIL → 0

    Правила:
    - Возвращай СТРОГО 1 или СТРОГО 0. Никаких промежуточных значений.
    - Нет "частично верно", "почти совпало", "семантически близко".
    - Нет оценок вероятности (0.7, 0.85) — только целое 0 или 1.
    - Если есть сомнения — выбирай 0 (как в TDD: красный тест).
    - Сравнение строковое и точное: регистр, пробелы, знаки — учитываются.
    - Если сравниваешь два списка, порядок элементов в списке НЕ ВАЖЕН. Например [a, б] = [б, а].
    - Если сравниваешь два списка, множество actual может содержать неполный список, это корректный ответ. Например если actual = [а] и expected = [а, б], то actual = expected.

    Твой score — это результат одного бинарного assert, не градация.
    """

    expected = dspy.InputField(desc="Ожидаемое значение (ground truth)")
    actual = dspy.InputField(desc="Предсказание модели")
    # reasoning = dspy.OutputField(desc="Краткое обоснование решения")
    score = dspy.OutputField(desc="1 если expected == actual, иначе 0")


def judge_metric_bin(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Возвращает 0 или 1"""
    expected = gold.expected
    actual = pred.predict

    with dspy.context(lm=judge_lm):
        res = dspy.Predict(JudgeSignature)(expected=expected, actual=actual)
        try:
            score = float(res.score)
        except (ValueError, TypeError):
            score = 0.0

    # для dspy.evaluate — достаточно float
    if pred_name is None:
        return score

    log.info(f"============actual: {actual}")
    log.info(f"============expected: {expected}")
    log.info(f"============score: {score}")

    # для GEPA-мутаций — текстовая обратная связь через reasoning judge-а
    # feedback = res.reasoning or (
    #     f"Ожидаемый ответ '{expected}' "
    #     f"Модель ответила '{actual}'."
    # )
    feedback = (
        f"expected '{expected}'. "
        f"actual '{actual}'. Промпт должен быть на русском языке. Не используй эмодзи."
    )
    return dspy.Prediction(score=float(score), feedback=feedback)


def judge_metric_float(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Метрика, совместимая с GEPA.

    Возвращает:
      - float               если pred_name is None (используется в dspy.evaluate);
      - dspy.Prediction     если pred_name задан (используется GEPA для мутаций).

    Шаблон взят из туториалов DSPy GEPA:
      - https://dspy.ai/tutorials/gepa_aime/
      - https://dspy.ai/tutorials/gepa_papillon/
    """
    expected = gold.expected
    actual = pred.predict

    expected_set = set(expected.replace(" ", "").split(","))
    actual_set = set(actual.replace(" ", "").split(","))

    if actual_set.issubset(expected_set):
        tp = len(expected_set & actual_set)
        score = tp / len(expected_set) if len(actual_set) > 0 else 0.0
    else:
        score = 0.0

    if pred_name is None:
        return score

    log.info(f"============actual: {actual}")
    log.info(f"============expected: {expected}")
    log.info(f"============score: {score}")

    feedback = f"expected '{expected}'. " f"actual '{actual}'."
    feedback += """
    НЕ используй лишние символы. Порядок лейблов не важен. Инструкция ТОЛЬКО на русском языке.
	"""
    return dspy.Prediction(score=float(score), feedback=feedback)


def _unescape(text: str) -> str:
    """
    Преобразует литеральные последовательности \n, \t, \u0410 … в реальные символы.
    Делается только при наличии обратного слеша, чтобы не «сломать» уже нормальный текст.
    """
    return codecs.decode(text, "unicode_escape") if "\\" in text else text


def _fix_mojibake(txt: str) -> str:
    """
    Если строка выглядит как mojibake (UTF‑8‑байты, интерпретированные как CP1251
    или KOI8‑R), возвращаем правильный русский.
    """
    # Признак mojibake – наличие символов типа Ð, Ñ, Â …
    if any(ch in txt for ch in ("Ð", "Ñ", "Â")):
        try:
            # CP1251 → UTF‑8
            return txt.encode("latin1").decode("utf-8")
        except UnicodeDecodeError:
            # иногда используется KOI8‑R
            return txt.encode("latin1").decode("koi8-r")
    return txt


def save_prompt(raw: Any, filename: str) -> None:
    """
    Сохраняет в *filename* поле «instruction», при необходимости
    - разбирает JSON,
    - разворачивает escape‑последовательности,
    - исправляет mojibake.
    Файл открывается в UTF‑8, поэтому в нём будет чистый русский текст.
    """
    # -------------------------------------------------
    # 1️⃣ Берём строку‑инструкцию
    # -------------------------------------------------
    if isinstance(raw, dict):
        instr = raw.get("instruction", "")
    elif isinstance(raw, str):
        s = raw.strip()
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None

        if isinstance(obj, dict):
            instr = obj.get("instruction", s)  # если поля нет – берём весь текст
        else:
            instr = s
    else:
        instr = str(raw)

    # -------------------------------------------------
    # 2️⃣ Очищаем строку
    # -------------------------------------------------
    if isinstance(instr, str):
        cleaned = _unescape(instr)  # \n, \uXXXX → реальные символы
        cleaned = _fix_mojibake(cleaned)  # поправка mojibake, если нужна
    else:
        cleaned = str(instr)

    # -------------------------------------------------
    # 3️⃣ Записываем в файл (UTF‑8)
    # -------------------------------------------------
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cleaned)

