import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
import matplotlib.pyplot as plt


class TargetBinarizer:
    """
    Класс для бинаризации целевой переменной.
    """

    def __init__(
            self,
            threshold: Optional[Union[int, float]] = None,
            threshold_strategy: str = 'fixed',
            percentile: float = 75.0,
            min_positive_ratio: float = 0.1,
            output_dir: Optional[str] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация бинаризатора целевой переменной.

        Args:
            threshold: Пороговое значение для бинаризации (если None, определяется автоматически)
            threshold_strategy: Стратегия определения порога ('fixed', 'percentile', 'auto')
            percentile: Перцентиль для стратегии 'percentile'
            min_positive_ratio: Минимальная доля положительного класса для стратегии 'auto'
            output_dir: Директория для сохранения визуализаций
            logger: Объект логгера
        """
        self.threshold = threshold
        self.threshold_strategy = threshold_strategy
        self.percentile = percentile
        self.min_positive_ratio = min_positive_ratio
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)

        self.optimized_threshold = None
        self.distribution_stats = {}

    def _determine_threshold(self, y: pd.Series) -> float:
        """
        Определение порогового значения для бинаризации.

        Args:
            y: Серия с целевой переменной

        Returns:
            float: Пороговое значение
        """
        if self.threshold_strategy == 'fixed' and self.threshold is not None:
            self.logger.info(f"Используется фиксированный порог: {self.threshold}")
            return self.threshold

        elif self.threshold_strategy == 'percentile':
            threshold = np.percentile(y, self.percentile)
            self.logger.info(f"Определен порог на основе {self.percentile}-го перцентиля: {threshold}")
            return threshold

        elif self.threshold_strategy == 'auto':
            # Ищем порог, который обеспечит не менее min_positive_ratio положительных примеров
            # Начинаем с медианы и двигаемся вниз
            threshold = np.median(y)
            positive_ratio = (y >= threshold).mean()

            while positive_ratio < self.min_positive_ratio and threshold > 0:
                threshold *= 0.9  # Уменьшаем порог на 10%
                positive_ratio = (y >= threshold).mean()

            self.logger.info(
                f"Автоматически определен порог: {threshold} (доля положительного класса: {positive_ratio:.2f})")
            return threshold

        else:
            # По умолчанию используем медиану
            threshold = np.median(y)
            self.logger.info(f"Используется порог по умолчанию (медиана): {threshold}")
            return threshold

    def _calculate_distribution_stats(self, y: pd.Series, y_binary: pd.Series) -> Dict:
        """
        Расчет статистики распределения значений целевой переменной.

        Args:
            y: Исходная целевая переменная
            y_binary: Бинаризованная целевая переменная

        Returns:
            Dict: Словарь со статистикой
        """
        # Общая статистика
        stats = {
            'threshold': self.optimized_threshold,
            'total_samples': len(y),
            'positive_samples': y_binary.sum(),
            'negative_samples': (1 - y_binary).sum(),
            'positive_ratio': y_binary.mean(),
            'negative_ratio': 1 - y_binary.mean(),
            'original_min': y.min(),
            'original_max': y.max(),
            'original_mean': y.mean(),
            'original_median': y.median(),
            'original_std': y.std()
        }

        # Статистика по классам
        stats['positive_class_min'] = y[y_binary == 1].min() if y_binary.sum() > 0 else None
        stats['positive_class_max'] = y[y_binary == 1].max() if y_binary.sum() > 0 else None
        stats['positive_class_mean'] = y[y_binary == 1].mean() if y_binary.sum() > 0 else None
        stats['positive_class_median'] = y[y_binary == 1].median() if y_binary.sum() > 0 else None

        stats['negative_class_min'] = y[y_binary == 0].min() if (1 - y_binary).sum() > 0 else None
        stats['negative_class_max'] = y[y_binary == 0].max() if (1 - y_binary).sum() > 0 else None
        stats['negative_class_mean'] = y[y_binary == 0].mean() if (1 - y_binary).sum() > 0 else None
        stats['negative_class_median'] = y[y_binary == 0].median() if (1 - y_binary).sum() > 0 else None

        return stats

    def _plot_distribution(self, y: pd.Series, y_binary: pd.Series) -> Optional[plt.Figure]:
        """
        Построение графика распределения целевой переменной.

        Args:
            y: Исходная целевая переменная
            y_binary: Бинаризованная целевая переменная

        Returns:
            Optional[plt.Figure]: Объект фигуры или None
        """
        if self.output_dir is None:
            return None

        # Создаем фигуру
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Гистограмма исходной целевой переменной
        ax1.hist(y, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.optimized_threshold, color='r', linestyle='--',
                    label=f'Порог: {self.optimized_threshold:.2f}')
        ax1.set_title('Распределение исходной целевой переменной')
        ax1.set_xlabel('Значение')
        ax1.set_ylabel('Частота')
        ax1.legend()

        # Гистограмма по классам
        positive_values = y[y_binary == 1]
        negative_values = y[y_binary == 0]

        if len(positive_values) > 0:
            ax2.hist(positive_values, bins=20, color='green', edgecolor='black',
                     alpha=0.5, label=f'Класс 1 ({len(positive_values)} примеров)')

        if len(negative_values) > 0:
            ax2.hist(negative_values, bins=20, color='red', edgecolor='black',
                     alpha=0.5, label=f'Класс 0 ({len(negative_values)} примеров)')

        ax2.axvline(self.optimized_threshold, color='r', linestyle='--')
        ax2.set_title('Распределение по классам')
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Частота')
        ax2.legend()

        plt.tight_layout()

        # Сохраняем визуализацию, если указана директория
        if self.output_dir:
            import os
            from pathlib import Path

            output_path = Path(self.output_dir) / 'target_distribution.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Визуализация распределения целевой переменной сохранена в {output_path}")

        return fig

    def fit(self, y: pd.Series) -> 'TargetBinarizer':
        """
        Обучение бинаризатора на целевой переменной.

        Args:
            y: Серия с целевой переменной

        Returns:
            TargetBinarizer: Обученный бинаризатор
        """
        # Проверяем на наличие пропусков
        if y.isna().any():
            self.logger.warning(f"Обнаружено {y.isna().sum()} пропусков в целевой переменной")

        # Определяем оптимальный порог
        self.optimized_threshold = self._determine_threshold(y.dropna())

        # Бинаризуем целевую переменную для сбора статистики
        y_binary = (y >= self.optimized_threshold).astype(int)

        # Рассчитываем статистику
        self.distribution_stats = self._calculate_distribution_stats(y, y_binary)

        # Логируем статистику
        self.logger.info(f"Порог бинаризации: {self.optimized_threshold}")
        self.logger.info(f"Доля положительного класса: {self.distribution_stats['positive_ratio']:.4f}")
        self.logger.info(f"Всего примеров: {self.distribution_stats['total_samples']}")
        self.logger.info(f"Положительных примеров: {self.distribution_stats['positive_samples']}")
        self.logger.info(f"Отрицательных примеров: {self.distribution_stats['negative_samples']}")

        # Строим и сохраняем визуализацию
        self._plot_distribution(y, y_binary)

        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Бинаризация целевой переменной.

        Args:
            y: Серия с целевой переменной

        Returns:
            pd.Series: Бинаризованная целевая переменная
        """
        if self.optimized_threshold is None:
            raise ValueError("Бинаризатор не обучен. Сначала вызовите метод fit()")

        # Бинаризуем целевую переменную
        y_binary = (y >= self.optimized_threshold).astype(int)

        # Обрабатываем пропуски
        if y.isna().any():
            self.logger.warning(f"Пропуски в целевой переменной заменены на 0 (отрицательный класс)")
            y_binary = y_binary.fillna(0).astype(int)

        return y_binary

    def fit_transform(self, y: pd.Series) -> pd.Series:
        """
        Обучение бинаризатора и бинаризация целевой переменной.

        Args:
            y: Серия с целевой переменной

        Returns:
            pd.Series: Бинаризованная целевая переменная
        """
        return self.fit(y).transform(y)

    def get_threshold(self) -> float:
        """
        Получение порогового значения для бинаризации.

        Returns:
            float: Пороговое значение
        """
        if self.optimized_threshold is None:
            raise ValueError("Бинаризатор не обучен. Сначала вызовите метод fit()")

        return self.optimized_threshold

    def get_stats(self) -> Dict:
        """
        Получение статистики распределения целевой переменной.

        Returns:
            Dict: Словарь со статистикой
        """
        if not self.distribution_stats:
            raise ValueError("Бинаризатор не обучен. Сначала вызовите метод fit()")

        return self.distribution_stats

    def get_quantiles(self, y: pd.Series, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]) -> Dict[
        float, float]:
        """
        Расчет квантилей целевой переменной.

        Args:
            y: Серия с целевой переменной
            quantiles: Список квантилей для расчета

        Returns:
            Dict[float, float]: Словарь с квантилями
        """
        return {q: float(np.percentile(y.dropna(), q * 100)) for q in quantiles}