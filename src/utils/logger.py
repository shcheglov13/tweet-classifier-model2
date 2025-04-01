import logging
import sys
from pathlib import Path
from typing import Union, Optional
import yaml


def setup_logger(
        log_file: Optional[Union[str, Path]] = None,
        level: str = 'INFO',
        name: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Настройка логгера с расширенной конфигурацией.

    Args:
        log_file: Путь к файлу логов
        level: Уровень логирования
        name: Имя логгера
        config_file: Путь к файлу конфигурации логгера

    Returns:
        logging.Logger: Настроенный логгер
    """
    # Получаем уровень логирования
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # Очищаем обработчики, если они уже были созданы

    # Применяем конфигурацию из файла, если она указана
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                logging.config.dictConfig(config)
                return logger
            except Exception as e:
                print(f"Ошибка при загрузке конфигурации логгера: {e}")

    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Создаем обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Создаем обработчик для файла, если путь указан
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_module_logger(name: str) -> logging.Logger:
    """
    Получение логгера для модуля.

    Args:
        name: Имя модуля (__name__)

    Returns:
        logging.Logger: Логгер для модуля
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Адаптер для логгера с дополнительным контекстом.
    """

    def __init__(self, logger, extra=None):
        """
        Инициализация адаптера.

        Args:
            logger: Базовый логгер
            extra: Дополнительный контекст
        """
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        """
        Обработка сообщения с добавлением контекста.

        Args:
            msg: Сообщение
            kwargs: Дополнительные параметры

        Returns:
            tuple: Обработанное сообщение и параметры
        """
        extra = self.extra.copy()
        context = kwargs.pop('extra', {})
        if context:
            extra.update(context)

        if extra:
            context_str = ' '.join(f"[{k}={v}]" for k, v in extra.items())
            msg = f"{msg} {context_str}"

        return msg, kwargs