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
	"""Инъекция для типовых классификаторов с различными инит промтами"""
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


def save_prompt(raw: str, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(raw)

