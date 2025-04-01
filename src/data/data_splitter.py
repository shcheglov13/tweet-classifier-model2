# src/data/data_splitter.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Класс для разделения данных на обучающую и тестовую выборки.
    """

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: int = 42,
            stratify: bool = True,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация сплиттера данных.

        Args:
            test_size: Доля данных для тестовой выборки
            random_state: Состояние генератора случайных чисел
            stratify: Стратифицировать ли выборки по целевой переменной
            logger: Объект логгера
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.logger = logger or logging.getLogger(__name__)

    def train_test_split(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Разделение данных на обучающую и тестовую выборки.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        self.logger.info(f"Разделение данных на обучающую и тестовую выборки (test_size={self.test_size})")

        stratify_param = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        self.logger.info(f"Размер обучающей выборки: {len(X_train)}")
        self.logger.info(f"Размер тестовой выборки: {len(X_test)}")

        if self.stratify:
            self.logger.info(f"Распределение классов в обучающей выборке: {dict(y_train.value_counts())}")
            self.logger.info(f"Распределение классов в тестовой выборке: {dict(y_test.value_counts())}")

        return X_train, X_test, y_train, y_test