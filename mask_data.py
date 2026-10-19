import re
from natasha import Doc, Segmenter, NewsEmbedding, NewsNERTagger


# Ленивая инициализация тяжёлых моделей natasha — грузятся один раз
_COMPONENTS = {}


def _get_components():
    if "segmenter" not in _COMPONENTS:
        _COMPONENTS["segmenter"] = Segmenter()
        _COMPONENTS["emb"] = NewsEmbedding()
        _COMPONENTS["ner_tagger"] = NewsNERTagger(_COMPONENTS["emb"])
    return _COMPONENTS["segmenter"], _COMPONENTS["ner_tagger"]


def replace_per_with_name(text: str) -> str:
    """Заменяет все сущности типа PER (Person) в тексте на плейсхолдер [NAME].

    Args:
        text: исходная строка на русском языке.

    Returns:
        Строка, в которой каждое распознанное ФИО заменено на '[NAME]'.
    """
    try:
        segmenter, ner_tagger = _get_components()

        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        # Идём с конца, чтобы индексы start/stop следующих спанов оставались валидными
        for span in reversed(doc.ner.spans):
            if span.type == "PER":
                text = text[: span.start] + "[NAME]" + text[span.stop :]
        return text
    except:
        return text


def replace_per_with_location(text: str) -> str:
    try:
        segmenter, ner_tagger = _get_components()

        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        # Идём с конца, чтобы индексы start/stop следующих спанов оставались валидными
        for span in reversed(doc.ner.spans):
            if span.type == "LOC":
                text = text[: span.start] + "[LOC]" + text[span.stop :]
        return text
    except:
        return text


def replace_per_with_organization(text: str) -> str:
    try:
        segmenter, ner_tagger = _get_components()

        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        # Идём с конца, чтобы индексы start/stop следующих спанов оставались валидными
        for span in reversed(doc.ner.spans):
            if span.type == "ORG":
                text = text[: span.start] + "[ORG]" + text[span.stop :]
        return text
    except:
        return text


def replace_emails(text: str) -> str:
    try:
        pattern = re.compile(r"\b\S+@\S+\b", re.IGNORECASE)
        return pattern.sub("[EMAIL]", text)
    except:
        return text


def mask_data(text):
    text = replace_emails(text)
    # text = replace_per_with_location(text)
    # text = replace_per_with_organization(text)
    text = replace_per_with_name(text)
    return text
