from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class FeatureExtractorBase(ABC):
    """
    Базовый абстрактный класс для всех экстракторов признаков.
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'FeatureExtractorBase':
        """
        Обучение экстрактора признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            self: Возвращает экземпляр класса
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование данных и извлечение признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd. DataFrame: Датафрейм с извлеченными признаками
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение экстрактора и извлечение признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd. DataFrame: Датафрейм с извлеченными признаками
        """
        return self.fit(data).transform(data)