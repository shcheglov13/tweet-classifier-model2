import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CorrelationSelector:
    """
    Класс для отбора признаков на основе корреляционного анализа.
    """

    def __init__(
            self,
            correlation_threshold: float = 0.95,
            embedding_prefixes: List[str] = ['text_emb_', 'quoted_text_emb_', 'image_emb_'],
            output_dir: Optional[Path] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация селектора.

        Args:
            correlation_threshold: Пороговое значение корреляции
            embedding_prefixes: Префиксы для идентификации колонок с эмбеддингами
            output_dir: Директория для сохранения визуализаций
            logger: Объект логгера
        """
        self.correlation_threshold = correlation_threshold
        self.embedding_prefixes = embedding_prefixes
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)

        self.features_to_drop = []
        self.correlation_matrix = None

    def _identify_embedding_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Идентификация колонок с эмбеддингами.

        Args:
            data: Датафрейм с признаками

        Returns:
            List[str]: Список колонок с эмбеддингами
        """
        embedding_columns = []
        for prefix in self.embedding_prefixes:
            embedding_columns.extend([col for col in data.columns if col.startswith(prefix)])

        return embedding_columns

    def _plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, output_path: Optional[Path] = None):
        """
        Построение тепловой карты корреляций.

        Args:
            correlation_matrix: Корреляционная матрица
            output_path: Путь для сохранения изображения
        """
        if correlation_matrix.shape[0] > 100:
            self.logger.info("Слишком много признаков для визуализации корреляции")
            return

        plt.figure(figsize=(16, 14))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title('Корреляционная матрица признаков')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Тепловая карта корреляций сохранена в {output_path}")

        plt.close()

    def fit(self, data: pd.DataFrame) -> 'CorrelationSelector':
        """
        Определение признаков с высокой корреляцией.

        Args:
            data: Датафрейм с признаками

        Returns:
            self: Возвращает экземпляр класса
        """
        self.logger.info(f"Начало корреляционного анализа с порогом {self.correlation_threshold}")

        # Идентифицируем эмбеддинги
        embedding_columns = self._identify_embedding_columns(data)

        # Анализируем только неэмбеддинговые признаки
        non_embedding_columns = [col for col in data.columns if col not in embedding_columns]

        if len(non_embedding_columns) <= 1:
            self.logger.info("Недостаточно признаков для корреляционного анализа")
            return self

        # Рассчитываем корреляционную матрицу
        self.correlation_matrix = data[non_embedding_columns].corr().abs()

        # Находим признаки с высокой корреляцией
        upper_tri = self.correlation_matrix.where(np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool))

        # Находим пары признаков с высокой корреляцией
        correlated_pairs = []
        for col in upper_tri.columns:
            high_corr_cols = upper_tri.index[upper_tri[col] > self.correlation_threshold]
            for high_corr_col in high_corr_cols:
                correlated_pairs.append((col, high_corr_col, upper_tri.loc[high_corr_col, col]))

        # Сортируем пары по убыванию корреляции
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)

        if correlated_pairs:
            self.logger.info(f"Найдено {len(correlated_pairs)} пар признаков с высокой корреляцией")
            for col1, col2, corr in correlated_pairs[:10]:  # Выводим первые 10 пар
                self.logger.info(f"Высокая корреляция ({corr:.4f}) между '{col1}' и '{col2}'")

        # Отбираем признаки для исключения, стараясь минимизировать количество удаляемых
        features_to_drop = set()
        for col1, col2, _ in correlated_pairs:
            # Добавляем к исключению признак, у которого больше других сильных корреляций
            corr_count1 = sum(1 for p in correlated_pairs if p[0] == col1 or p[1] == col1)
            corr_count2 = sum(1 for p in correlated_pairs if p[0] == col2 or p[1] == col2)

            if corr_count1 > corr_count2:
                features_to_drop.add(col1)
            else:
                features_to_drop.add(col2)

        self.features_to_drop = list(features_to_drop)

        self.logger.info(f"Выбрано {len(self.features_to_drop)} признаков для исключения")

        # Визуализация корреляционной матрицы
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._plot_correlation_heatmap(
                self.correlation_matrix,
                self.output_dir / 'correlation_heatmap.png'
            )

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Исключение признаков с высокой корреляцией.

        Args:
            data: Датафрейм с признаками

        Returns:
            pd.DataFrame: Датафрейм с отобранными признаками
        """
        if not self.features_to_drop:
            self.logger.info("Нет признаков для исключения")
            return data

        self.logger.info(f"Исключение {len(self.features_to_drop)} признаков с высокой корреляцией")

        # Оставляем только те признаки, которые есть в данных
        cols_to_drop = [col for col in self.features_to_drop if col in data.columns]

        if cols_to_drop:
            result = data.drop(columns=cols_to_drop)
            self.logger.info(f"Исключено {len(cols_to_drop)} признаков. Осталось {result.shape[1]} признаков")
            return result
        else:
            return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Определение и исключение признаков с высокой корреляцией.

        Args:
            data: Датафрейм с признаками

        Returns:
            pd.DataFrame: Датафрейм с отобранными признаками
        """
        return self.fit(data).transform(data)

    def get_correlation_pairs(self, top_n: int = 20) -> List[Dict]:
        """
        Получение пар признаков с высокой корреляцией.

        Args:
            top_n: Количество пар для возврата

        Returns:
            List[Dict]: Список словарей с парами признаков и их корреляциями
        """
        if self.correlation_matrix is None:
            return []

        # Находим пары признаков с наибольшей корреляцией
        pairs = []
        upper_tri = self.correlation_matrix.where(np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool))

        for col in upper_tri.columns:
            for idx in upper_tri.index:
                corr_value = upper_tri.loc[idx, col]
                if not np.isnan(corr_value):
                    pairs.append({
                        'feature1': col,
                        'feature2': idx,
                        'correlation': corr_value
                    })

        # Сортируем по убыванию корреляции
        pairs.sort(key=lambda x: x['correlation'], reverse=True)

        return pairs[:top_n]