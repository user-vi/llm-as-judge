
import logging
from pathlib import Path


def setup_dspy_logger(
    log_file: str | Path = "./gepa.log",
    *,
    stdout_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    logger_name: str = "dspy",
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    mode: str = "w",
) -> logging.Logger:
    """Сконфигурировать логгер DSPy: файл + stdout + фильтр.

    Параметры:
        log_file: путь к лог-файлу; родительская директория создаётся автоматически.
        stdout_level: уровень для StreamHandler (по умолчанию INFO).
        file_level: уровень для FileHandler (по умолчанию DEBUG — пишем всё).
        logger_name: имя логгера (по умолчанию "dspy" — корневой для всей библиотеки).
        fmt / datefmt: формат и формат даты для Format.
        filter_keywords: ключевые слова, по которым фильтруем INFO-сообщения.
        mode: режим открытия файла ("w" — перезаписывать, "a" — дописывать).

    Возвращает:
        Настроенный логгер.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode=mode)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    # file_handler.addFilter(keyword_filter)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)
    # stdout_handler.addFilter(keyword_filter)

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()          # убираем дефолтные хэндлеры DSPy
    logger.setLevel(logging.DEBUG)    # головной уровень — самый низкий
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.propagate = False          # не дублировать в root

    logger.info("DSPy logger initialised: file=%s, stdout_level=%s",
                log_path.resolve(), logging.getLevelName(stdout_level))
    return logger
