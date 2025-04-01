import pandas as pd
import numpy as np
import logging
from typing import  List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


class FeatureScaler:
    """
    Класс для масштабирования признаков.
    """

    def __init__(
            self,
            scaler_type: str = 'standard',
            scale_embeddings: bool = False,
            embedding_prefixes: List[str] = ['text_emb_', 'quoted_text_emb_', 'image_emb_'],
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация масштабировщика признаков.

        Args:
            scaler_type: Тип масштабирования ('standard', 'minmax', 'robust', 'maxabs')
            scale_embeddings: Масштабировать ли эмбеддинги
            embedding_prefixes: Префиксы для идентификации колонок с эмбеддингами
            logger: Объект логгера
        """
        self.scaler_type = scaler_type
        self.scale_embeddings = scale_embeddings
        self.embedding_prefixes = embedding_prefixes
        self.logger = logger or logging.getLogger(__name__)

        # Инициализируем масштабировщик в зависимости от типа
        self.scaler = self._create_scaler(scaler_type)

        # Списки для хранения признаков разных типов
        self.embedding_columns = []
        self.feature_columns = []

    def _create_scaler(self, scaler_type: str):
        """
        Создание масштабировщика нужного типа.

        Args:
            scaler_type: Тип масштабирования

        Returns:
            Масштабировщик
        """
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'maxabs':
            return MaxAbsScaler()
        else:
            self.logger.warning(f"Неизвестный тип масштабирования: {scaler_type}. Используем StandardScaler")
            return StandardScaler()

    def _identify_column_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Определение типов колонок (эмбеддинги и другие признаки).

        Args:
            data: Датафрейм с данными

        Returns:
            Tuple[List[str], List[str]]: Списки с именами колонок по типам
        """
        embedding_cols = []
        feature_cols = []

        for col in data.columns:
            # Проверяем, является ли колонка эмбеддингом по префиксу
            is_embedding = any(col.startswith(prefix) for prefix in self.embedding_prefixes)

            if is_embedding:
                embedding_cols.append(col)
            else:
                # Проверяем, является ли колонка числовой
                if pd.api.types.is_numeric_dtype(data[col]):
                    feature_cols.append(col)

        self.logger.info(f"Определено {len(embedding_cols)} колонок с эмбеддингами")
        self.logger.info(f"Определено {len(feature_cols)} числовых признаков для масштабирования")

        return embedding_cols, feature_cols

    def fit(self, data: pd.DataFrame) -> 'FeatureScaler':
        """
        Обучение масштабировщика на данных.

        Args:
            data: Датафрейм с данными

        Returns:
            FeatureScaler: Обученный масштабировщик
        """
        self.embedding_columns, self.feature_columns = self._identify_column_types(data)

        # Если нет числовых признаков, нечего масштабировать
        if not self.feature_columns and not (self.scale_embeddings and self.embedding_columns):
            self.logger.warning("Нет числовых признаков для масштабирования")
            return self

        # Определяем, какие колонки масштабировать
        columns_to_scale = []

        if self.feature_columns:
            columns_to_scale.extend(self.feature_columns)

        if self.scale_embeddings and self.embedding_columns:
            columns_to_scale.extend(self.embedding_columns)

        # Проверяем на наличие NaN и бесконечных значений
        if data[columns_to_scale].isna().any().any() or np.isinf(data[columns_to_scale].values).any():
            self.logger.warning(
                "Обнаружены пропуски или бесконечные значения. Рекомендуется обработать их перед масштабированием.")

        # Обучаем масштабировщик
        self.logger.info(f"Обучение {self.scaler_type} масштабировщика на {len(columns_to_scale)} признаках")
        self.scaler.fit(data[columns_to_scale])

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Масштабирование признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с масштабированными признаками
        """
        # Если нет числовых признаков или масштабировщик не обучен, возвращаем исходные данные
        if not hasattr(self, 'feature_columns') or (
                not self.feature_columns and not (self.scale_embeddings and self.embedding_columns)):
            return data.copy()

        # Определяем, какие колонки масштабировать
        columns_to_scale = []

        if self.feature_columns:
            # Проверяем, что все колонки для масштабирования присутствуют в данных
            feature_cols_present = [col for col in self.feature_columns if col in data.columns]
            if len(feature_cols_present) < len(self.feature_columns):
                self.logger.warning(
                    f"Не все признаки для масштабирования присутствуют в данных. Отсутствует {len(self.feature_columns) - len(feature_cols_present)} признаков.")

            columns_to_scale.extend(feature_cols_present)

        if self.scale_embeddings and self.embedding_columns:
            # Проверяем, что все колонки эмбеддингов присутствуют в данных
            embedding_cols_present = [col for col in self.embedding_columns if col in data.columns]
            if len(embedding_cols_present) < len(self.embedding_columns):
                self.logger.warning(
                    f"Не все эмбеддинги присутствуют в данных. Отсутствует {len(self.embedding_columns) - len(embedding_cols_present)} колонок.")

            columns_to_scale.extend(embedding_cols_present)

        # Если нет колонок для масштабирования, возвращаем исходные данные
        if not columns_to_scale:
            self.logger.warning("Нет колонок для масштабирования в данных")
            return data.copy()

        # Копируем исходные данные
        transformed_data = data.copy()

        # Проверяем на наличие NaN и бесконечных значений
        if transformed_data[columns_to_scale].isna().any().any() or np.isinf(
                transformed_data[columns_to_scale].values).any():
            self.logger.warning(
                "Обнаружены пропуски или бесконечные значения. Результаты масштабирования могут быть некорректными.")

        # Масштабируем признаки
        self.logger.info(f"Масштабирование {len(columns_to_scale)} признаков")
        transformed_data[columns_to_scale] = self.scaler.transform(transformed_data[columns_to_scale])

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение масштабировщика и масштабирование признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с масштабированными признаками
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обратное масштабирование признаков.

        Args:
            data: Датафрейм с масштабированными признаками

        Returns:
            pd.DataFrame: Датафрейм с исходными значениями признаков
        """
        # Если масштабировщик не обучен, возвращаем исходные данные
        if not hasattr(self, 'feature_columns') or (
                not self.feature_columns and not (self.scale_embeddings and self.embedding_columns)):
            return data.copy()

        # Определяем, какие колонки обратно масштабировать
        columns_to_scale = []

        if self.feature_columns:
            feature_cols_present = [col for col in self.feature_columns if col in data.columns]
            columns_to_scale.extend(feature_cols_present)

        if self.scale_embeddings and self.embedding_columns:
            embedding_cols_present = [col for col in self.embedding_columns if col in data.columns]
            columns_to_scale.extend(embedding_cols_present)

        # Если нет колонок для масштабирования, возвращаем исходные данные
        if not columns_to_scale:
            return data.copy()

        # Копируем исходные данные
        inverse_data = data.copy()

        # Обратное масштабирование
        self.logger.info(f"Обратное масштабирование {len(columns_to_scale)} признаков")
        inverse_data[columns_to_scale] = self.scaler.inverse_transform(inverse_data[columns_to_scale])

        return inverse_data

    def get_feature_ranges(self) -> pd.DataFrame:
        """
        Получение диапазонов значений признаков до и после масштабирования.

        Returns:
            pd.DataFrame: Датафрейм с диапазонами значений
        """
        if not hasattr(self, 'feature_columns') or not hasattr(self.scaler, 'scale_'):
            return pd.DataFrame()

        # Собираем данные о масштабировании
        columns_to_scale = []

        if self.feature_columns:
            columns_to_scale.extend(self.feature_columns)

        if self.scale_embeddings and self.embedding_columns:
            columns_to_scale.extend(self.embedding_columns)

        # Для разных типов масштабировщиков разные параметры
        if isinstance(self.scaler, StandardScaler):
            scale_data = {
                'column': columns_to_scale,
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_,
                'var': self.scaler.var_
            }
        elif isinstance(self.scaler, MinMaxScaler):
            scale_data = {
                'column': columns_to_scale,
                'min': self.scaler.data_min_,
                'max': self.scaler.data_max_,
                'scale': self.scaler.scale_,
                'min_scaled': self.scaler.feature_range[0],
                'max_scaled': self.scaler.feature_range[1]
            }
        elif isinstance(self.scaler, RobustScaler):
            scale_data = {
                'column': columns_to_scale,
                'center': self.scaler.center_,
                'scale': self.scaler.scale_
            }
        elif isinstance(self.scaler, MaxAbsScaler):
            scale_data = {
                'column': columns_to_scale,
                'max_abs': self.scaler.max_abs_,
                'scale': self.scaler.scale_
            }
        else:
            scale_data = {
                'column': columns_to_scale,
                'scale': self.scaler.scale_
            }

        return pd.DataFrame(scale_data)