import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer


class MissingValuesHandler:
    """
    Класс для обработки пропущенных значений в данных.
    """

    def __init__(
            self,
            strategy: str = 'zeros',
            categorical_strategy: str = 'most_frequent',
            numeric_strategy: str = 'mean',
            embedding_strategy: str = 'zeros',
            knn_neighbors: int = 5,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация обработчика пропущенных значений.

        Args:
            strategy: Общая стратегия обработки пропусков ('zeros', 'mean', 'median', 'most_frequent', 'knn', 'keep_nan')
            categorical_strategy: Стратегия для категориальных признаков
            numeric_strategy: Стратегия для числовых признаков
            embedding_strategy: Стратегия для эмбеддингов
            knn_neighbors: Количество соседей для KNN-заполнения
            logger: Объект логгера
        """
        self.strategy = strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_strategy = numeric_strategy
        self.embedding_strategy = embedding_strategy
        self.knn_neighbors = knn_neighbors
        self.logger = logger or logging.getLogger(__name__)

        self.numeric_imputer = None
        self.categorical_imputer = None
        self.embedding_imputer = None

        self.numeric_columns = []
        self.categorical_columns = []
        self.embedding_columns = []

    def _identify_column_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Определение типов колонок (числовые, категориальные, эмбеддинги).

        Args:
            data: Датафрейм с данными

        Returns:
            Tuple[List[str], List[str], List[str]]: Списки с именами колонок по типам
        """
        numeric_cols = []
        categorical_cols = []
        embedding_cols = []

        for col in data.columns:
            # Эмбеддинги определяем по имени
            if 'emb_' in col:
                embedding_cols.append(col)
            # Категориальные - по типу данных
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                categorical_cols.append(col)
            # Остальные считаем числовыми
            else:
                numeric_cols.append(col)

        self.logger.info(f"Определено {len(numeric_cols)} числовых колонок")
        self.logger.info(f"Определено {len(categorical_cols)} категориальных колонок")
        self.logger.info(f"Определено {len(embedding_cols)} колонок с эмбеддингами")

        return numeric_cols, categorical_cols, embedding_cols

    def fit(self, data: pd.DataFrame) -> 'MissingValuesHandler':
        """
        Обучение обработчика пропущенных значений.

        Args:
            data: Датафрейм с данными

        Returns:
            MissingValuesHandler: Обученный обработчик
        """
        self.numeric_columns, self.categorical_columns, self.embedding_columns = self._identify_column_types(data)

        # Если выбрана стратегия сохранения NaN, не создаем импьютеры
        if self.strategy == 'keep_nan':
            self.logger.info("Выбрана стратегия сохранения NaN, пропуски не будут заполнены")
            return self

        # Выбираем стратегии для каждого типа данных в соответствии с общей стратегией
        if self.strategy != 'zeros' and self.strategy != 'knn':
            numeric_strategy = self.strategy if self.numeric_strategy is None else self.numeric_strategy
            categorical_strategy = self.strategy if self.categorical_strategy is None else self.categorical_strategy
            embedding_strategy = self.strategy if self.embedding_strategy is None else self.embedding_strategy
        else:
            numeric_strategy = self.numeric_strategy
            categorical_strategy = self.categorical_strategy
            embedding_strategy = self.embedding_strategy

        # Создаем импьютеры в зависимости от стратегии
        if self.strategy == 'knn':
            self.logger.info(f"Создание KNN-импьютера с {self.knn_neighbors} соседями")
            if self.numeric_columns:
                self.numeric_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                # Обучаем на числовых данных
                self.numeric_imputer.fit(data[self.numeric_columns])

            # Для категориальных данных KNN не подходит, используем most_frequent
            if self.categorical_columns:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                self.categorical_imputer.fit(data[self.categorical_columns])

            # Для эмбеддингов тоже используем KNN
            if self.embedding_columns:
                self.embedding_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                self.embedding_imputer.fit(data[self.embedding_columns])

        elif self.strategy == 'zeros':
            self.logger.info("Выбрана стратегия заполнения нулями")
            # Для всех типов данных будем использовать заполнение нулями
            # Не требуется обучения импьютеров

        else:
            # Обычная стратегия с SimpleImputer
            if self.numeric_columns:
                self.logger.info(f"Создание импьютера для числовых признаков со стратегией {numeric_strategy}")
                self.numeric_imputer = SimpleImputer(strategy=numeric_strategy)
                self.numeric_imputer.fit(data[self.numeric_columns])

            if self.categorical_columns:
                self.logger.info(
                    f"Создание импьютера для категориальных признаков со стратегией {categorical_strategy}")
                self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
                self.categorical_imputer.fit(data[self.categorical_columns])

            if self.embedding_columns:
                self.logger.info(f"Создание импьютера для эмбеддингов со стратегией {embedding_strategy}")
                self.embedding_imputer = SimpleImputer(strategy=embedding_strategy)
                self.embedding_imputer.fit(data[self.embedding_columns])

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Заполнение пропущенных значений в данных.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с заполненными пропусками
        """
        # Если нет данных или выбрана стратегия сохранения NaN, возвращаем исходный датафрейм
        if data.empty or self.strategy == 'keep_nan':
            return data.copy()

        # Копируем исходные данные
        transformed_data = data.copy()

        # Если выбрана стратегия заполнения нулями
        if self.strategy == 'zeros':
            self.logger.info("Заполнение пропусков нулями")
            transformed_data = transformed_data.fillna(0)
            return transformed_data

        # Обрабатываем каждый тип данных отдельно
        if self.numeric_columns and self.numeric_imputer is not None:
            self.logger.info("Заполнение пропусков в числовых признаках")
            numeric_data = transformed_data[self.numeric_columns]

            # Проверяем, есть ли пропуски
            if numeric_data.isna().any().any():
                numeric_filled = self.numeric_imputer.transform(numeric_data)
                transformed_data[self.numeric_columns] = numeric_filled

        if self.categorical_columns and self.categorical_imputer is not None:
            self.logger.info("Заполнение пропусков в категориальных признаках")
            categorical_data = transformed_data[self.categorical_columns]

            # Проверяем, есть ли пропуски
            if categorical_data.isna().any().any():
                categorical_filled = self.categorical_imputer.transform(categorical_data)
                transformed_data[self.categorical_columns] = categorical_filled

        if self.embedding_columns and self.embedding_imputer is not None:
            self.logger.info("Заполнение пропусков в эмбеддингах")
            embedding_data = transformed_data[self.embedding_columns]

            # Проверяем, есть ли пропуски
            if embedding_data.isna().any().any():
                embedding_filled = self.embedding_imputer.transform(embedding_data)
                transformed_data[self.embedding_columns] = embedding_filled

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение обработчика и заполнение пропущенных значений.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с заполненными пропусками
        """
        return self.fit(data).transform(data)

    def get_missing_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Получение статистики по пропущенным значениям.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Статистика по пропущенным значениям
        """
        # Рассчитываем количество и процент пропусков по каждой колонке
        missing_count = data.isna().sum()
        missing_percent = (data.isna().sum() / len(data) * 100).round(2)

        # Создаем датафрейм со статистикой
        missing_stats = pd.DataFrame({
            'column': missing_count.index,
            'missing_count': missing_count.values,
            'missing_percent': missing_percent.values
        })

        # Сортируем по убыванию количества пропусков
        missing_stats = missing_stats.sort_values('missing_count', ascending=False)

        # Добавляем тип данных и тип колонки
        missing_stats['dtype'] = missing_stats['column'].apply(lambda col: str(data[col].dtype))

        def get_column_type(col):
            if col in self.numeric_columns:
                return 'numeric'
            elif col in self.categorical_columns:
                return 'categorical'
            elif col in self.embedding_columns:
                return 'embedding'
            else:
                return 'unknown'

        missing_stats['column_type'] = missing_stats['column'].apply(get_column_type)

        return missing_stats