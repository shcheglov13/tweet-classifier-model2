import logging
import time
from typing import Optional, Union, Any
from tqdm.auto import tqdm


class ProgressTracker:
    """
    Класс для отслеживания прогресса операций с разумным логированием и tqdm.
    Ограничивает количество сообщений в логах, выводя только значимые изменения.
    """

    def __init__(
            self,
            total: int,
            description: str = "Processing",
            logger: Optional[logging.Logger] = None,
            log_interval: float = 50.0,
            disable_tqdm: bool = False,
            **tqdm_kwargs: Any
    ):
        """
        Инициализация трекера прогресса.

        Args:
            total: Общее количество элементов для обработки
            description: Описание операции
            logger: Объект логгера
            log_interval: Интервал логирования в процентах (по умолчанию каждые 10%)
            disable_tqdm: Отключить отображение tqdm в консоли
            **tqdm_kwargs: Дополнительные аргументы для tqdm
        """
        self.logger = logger or logging.getLogger(__name__)
        self.description = description
        self.total = max(1, total)  # Защита от деления на ноль
        self.log_interval = log_interval
        self.last_logged_percentage = 0
        self.start_time = time.time()

        # Инициализация tqdm для вывода в консоль
        kwargs = {
            'total': total,
            'desc': description,
            'disable': disable_tqdm,
            'unit': 'it',
            'ncols': 100
        }
        kwargs.update(tqdm_kwargs)
        self.pbar = tqdm(**kwargs)

        # Начальное сообщение в логах
        self.logger.info(f"Начало: {description} (всего элементов: {total})")

    def update(self, increment: int = 1) -> None:
        """
        Обновление прогресса.

        Args:
            increment: Количество обработанных элементов
        """
        # Обновляем tqdm для отображения в консоли
        self.pbar.update(increment)

        # Проверяем, нужно ли логировать
        current = min(self.pbar.n, self.total)
        current_percentage = int((current / self.total) * 100)

        # Логируем только если процент изменился существенно или достигли 100%
        if (current_percentage >= self.last_logged_percentage + self.log_interval) or (current >= self.total):
            self._log_progress()
            self.last_logged_percentage = current_percentage

    def _log_progress(self) -> None:
        """
        Логирование текущего прогресса.
        """
        current = min(self.pbar.n, self.total)
        percentage = int((current / self.total) * 100)

        # Расчет времени
        elapsed_time = time.time() - self.start_time
        if current > 0:
            estimated_total_time = elapsed_time * self.total / current
            estimated_remaining_time = estimated_total_time - elapsed_time
        else:
            estimated_remaining_time = 0

        # Форматирование времени
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(estimated_remaining_time)

        # Логируем прогресс
        status = f"{self.description}: {percentage}% завершено ({current}/{self.total}) - Прошло: {elapsed_str}, Осталось: {remaining_str}"
        self.logger.info(status)

    def _format_time(self, seconds: float) -> str:
        """
        Форматирование времени в человекочитаемый вид.

        Args:
            seconds: Время в секундах

        Returns:
            str: Отформатированное время
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours}ч {minutes}м {seconds}с"
        elif minutes > 0:
            return f"{minutes}м {seconds}с"
        else:
            return f"{seconds}с"

    def set_description(self, desc: str) -> None:
        """
        Обновление описания прогресса.

        Args:
            desc: Новое описание
        """
        self.pbar.set_description(desc)
        self.description = desc
        self.logger.info(f"Этап: {desc}")

    def set_postfix(self, **kwargs: Union[str, int, float]) -> None:
        """
        Установка дополнительной информации.

        Args:
            **kwargs: Ключи и значения для отображения
        """
        self.pbar.set_postfix(**kwargs)

    def complete(self) -> None:
        """
        Завершение отслеживания прогресса.
        """
        self.pbar.close()
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"{self.description} завершено за {self._format_time(elapsed_time)}")

    def __enter__(self) -> 'ProgressTracker':
        """
        Поддержка контекстного менеджера.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Закрытие прогресс-бара при выходе из контекста.
        """
        self.complete()