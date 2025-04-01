import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    Класс для предобработки данных перед обучением моделей.
    """

    def __init__(
            self,
            missing_strategy: str = 'zeros',
            scaler_type: str = 'standard',
            target_threshold: int = 100,
            target_column: str = 'tx_count',
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация препроцессора данных.

        Args:
            missing_strategy: Стратегия обработки пропусков ('zeros' или 'keep_nan')
            scaler_type: Тип масштабирования ('standard' или 'minmax')
            target_threshold: Порог для бинаризации целевой переменной
            target_column: Название колонки с целевой переменной
            logger: Объект логгера
        """
        self.missing_strategy = missing_strategy
        self.scaler_type = scaler_type
        self.target_threshold = target_threshold
        self.target_column = target_column
        self.logger = logger or logging.getLogger(__name__)

        self.scaler = None
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Неподдерживаемый тип масштабирования: {scaler_type}")

        self.embedding_columns = []
        self.numerical_columns = []

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка пропущенных значений.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Обработанный датафрейм
        """
        if self.missing_strategy == 'zeros':
            self.logger.info("Заполнение пропусков нулями")
            return data.fillna(0)
        elif self.missing_strategy == 'keep_nan':
            self.logger.info("Сохранение пропущенных значений")
            return data
        else:
            raise ValueError(f"Неподдерживаемая стратегия обработки пропусков: {self.missing_strategy}")

    def _identify_column_types(self, data: pd.DataFrame):
        """
        Определение типов колонок (эмбеддинги, числовые).

        Args:
            data: Датафрейм с данными
        """
        self.embedding_columns = [col for col in data.columns if ('emb_' in col)]
        self.numerical_columns = [
            col for col in data.columns
            if col not in self.embedding_columns and col != self.target_column
        ]

        self.logger.info(f"Определено {len(self.embedding_columns)} колонок с эмбеддингами")
        self.logger.info(f"Определено {len(self.numerical_columns)} числовых колонок")

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Масштабирование признаков.

        Args:
            data: Датафрейм с признаками

        Returns:
            pd.DataFrame: Датафрейм с масштабированными признаками
        """
        self.logger.info(f"Масштабирование признаков с использованием {self.scaler_type}")

        # Масштабируем только числовые колонки
        if len(self.numerical_columns) > 0:
            data_scaled = data.copy()
            data_scaled[self.numerical_columns] = self.scaler.transform(data[self.numerical_columns])
            return data_scaled
        else:
            return data

    def _binarize_target(self, y: pd.Series) -> pd.Series:
        """
        Бинаризация целевой переменной.

        Args:
            y: Серия с целевой переменной

        Returns:
            pd.Series: Бинаризованная целевая переменная
        """
        self.logger.info(f"Бинаризация целевой переменной с порогом {self.target_threshold}")
        y_binary = (y >= self.target_threshold).astype(int)

        # Логируем распределение классов
        class_counts = y_binary.value_counts()
        class_percentages = class_counts / len(y_binary) * 100

        for cls, count in class_counts.items():
            self.logger.info(f"Класс {cls}: {count} примеров ({class_percentages[cls]:.2f}%)")

        return y_binary

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Обучение препроцессора.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной

        Returns:
            self: Возвращает экземпляр класса
        """
        self._identify_column_types(X)

        # Обработка пропущенных значений
        X_processed = self._handle_missing_values(X)

        # Обучаем скалер на числовых признаках
        if len(self.numerical_columns) > 0:
            self.logger.info(f"Обучение скалера на {len(self.numerical_columns)} числовых признаках")
            self.scaler.fit(X_processed[self.numerical_columns])

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[
        pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Преобразование данных.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной (опционально)

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]: Преобразованные данные
        """
        # Обработка пропущенных значений
        X_processed = self._handle_missing_values(X)

        # Масштабирование признаков
        X_scaled = self._scale_features(X_processed)

        if y is not None:
            # Бинаризация целевой переменной
            y_binary = self._binarize_target(y)
            return X_scaled, y_binary
        else:
            return X_scaled

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[
        pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Обучение препроцессора и преобразование данных.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной (опционально)

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]: Преобразованные данные
        """
        return self.fit(X, y).transform(X, y)