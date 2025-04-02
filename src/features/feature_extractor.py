import pandas as pd
import numpy as np
import logging
from typing import Optional, Union
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.features.feature_base import FeatureExtractorBase
from src.features.text_features import TextFeatureExtractor
from src.features.image_features import ImageFeatureExtractor
from src.features.struct_features import StructuralFeatureExtractor


class FeatureExtractor(FeatureExtractorBase):
    """
    Общий экстрактор признаков, объединяющий текстовые, визуальные и структурные признаки.
    """

    def __init__(
            self,
            cache_dir: Optional[Union[str, Path]] = None,
            dimension_reduction: str = 'pca',
            n_components: int = 50,
            batch_size: int = 32,
            device: Optional[str] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация общего экстрактора признаков.

        Args:
            cache_dir: Директория для кеширования
            dimension_reduction: Метод снижения размерности ('pca' или 'tsne')
            n_components: Количество компонент после снижения размерности
            batch_size: Размер батча для обработки
            device: Устройство для вычислений ('cpu' или 'cuda')
            logger: Объект логгера
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.dimension_reduction = dimension_reduction
        self.n_components = n_components
        self.batch_size = batch_size
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Инициализация экстракторов для каждого типа признаков
        self.text_extractor = TextFeatureExtractor(
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
            device=self.device,
            logger=self.logger
        )

        self.image_extractor = ImageFeatureExtractor(
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
            device=self.device,
            logger=self.logger
        )

        self.struct_extractor = StructuralFeatureExtractor(
            cache_dir=self.cache_dir,
            logger=self.logger
        )

        # Модели для снижения размерности
        self.text_dim_reducer = None
        self.quoted_text_dim_reducer = None
        self.image_dim_reducer = None

    def _create_dim_reducer(self):
        """Создание модели для снижения размерности."""
        if self.dimension_reduction == 'pca':
            return PCA(n_components=self.n_components, random_state=42)
        elif self.dimension_reduction == 'tsne':
            return TSNE(n_components=self.n_components, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Неподдерживаемый метод снижения размерности: {self.dimension_reduction}")

    def _reduce_dimensions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Снижение размерности для эмбеддингов.

        Args:
            features_df: Датафрейм с признаками

        Returns:
            pd.DataFrame: Датафрейм с сокращенной размерностью эмбеддингов
        """
        self.logger.info(f"Снижение размерности эмбеддингов методом {self.dimension_reduction}")

        result_df = features_df.copy()

        # Идентификация колонок с эмбеддингами для каждого типа
        text_emb_cols = [col for col in features_df.columns if col.startswith('text_emb_')]
        quoted_text_emb_cols = [col for col in features_df.columns if col.startswith('quoted_text_emb_')]
        image_emb_cols = [col for col in features_df.columns if col.startswith('image_emb_')]

        # Снижение размерности для текстовых эмбеддингов
        if text_emb_cols and len(text_emb_cols) > self.n_components:
            text_embeddings = features_df[text_emb_cols].values
            reduced_text_embeddings = self._reduce_dimensions_for_embeddings(text_embeddings, 'text')

            # Удаляем исходные колонки и добавляем новые
            result_df = result_df.drop(columns=text_emb_cols)
            for i in range(reduced_text_embeddings.shape[1]):
                result_df[f'text_emb_reduced_{i}'] = reduced_text_embeddings[:, i]

            self.logger.info(
                f"Размерность текстовых эмбеддингов сокращена с {len(text_emb_cols)} до {reduced_text_embeddings.shape[1]}")

        # Снижение размерности для эмбеддингов цитируемого текста
        if quoted_text_emb_cols and len(quoted_text_emb_cols) > self.n_components:
            quoted_text_embeddings = features_df[quoted_text_emb_cols].values
            reduced_quoted_text_embeddings = self._reduce_dimensions_for_embeddings(quoted_text_embeddings,
                                                                                    'quoted_text')

            # Удаляем исходные колонки и добавляем новые
            result_df = result_df.drop(columns=quoted_text_emb_cols)
            for i in range(reduced_quoted_text_embeddings.shape[1]):
                result_df[f'quoted_text_emb_reduced_{i}'] = reduced_quoted_text_embeddings[:, i]

            self.logger.info(
                f"Размерность эмбеддингов цитируемого текста сокращена с {len(quoted_text_emb_cols)} до {reduced_quoted_text_embeddings.shape[1]}")

        # Снижение размерности для эмбеддингов изображений
        if image_emb_cols and len(image_emb_cols) > self.n_components:
            image_embeddings = features_df[image_emb_cols].values
            reduced_image_embeddings = self._reduce_dimensions_for_embeddings(image_embeddings, 'image')

            # Удаляем исходные колонки и добавляем новые
            result_df = result_df.drop(columns=image_emb_cols)
            for i in range(reduced_image_embeddings.shape[1]):
                result_df[f'image_emb_reduced_{i}'] = reduced_image_embeddings[:, i]

            self.logger.info(
                f"Размерность эмбеддингов изображений сокращена с {len(image_emb_cols)} до {reduced_image_embeddings.shape[1]}")

        return result_df


    def _reduce_dimensions_for_embeddings(self, embeddings: np.ndarray, embedding_type: str) -> np.ndarray:
        """
        Снижение размерности для определенного типа эмбеддингов.

        Args:
            embeddings: Матрица эмбеддингов
            embedding_type: Тип эмбеддингов ('text', 'quoted_text', 'image')

        Returns:
            np.ndarray: Матрица с сокращенной размерностью
        """
        # Получаем соответствующий редьюсер для типа эмбеддингов
        dim_reducer_attr = f"{embedding_type}_dim_reducer"

        # Создаем редьюсер, если его еще нет
        if getattr(self, dim_reducer_attr, None) is None:
            setattr(self, dim_reducer_attr, self._create_dim_reducer())
            getattr(self, dim_reducer_attr).fit(embeddings)

        # Применяем снижение размерности
        return getattr(self, dim_reducer_attr).transform(embeddings)

    def fit(self, data: pd.DataFrame) -> 'FeatureExtractor':
        """
        Обучение всех экстракторов признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            self: Возвращает экземпляр класса
        """
        self.logger.info("Обучение экстракторов признаков")

        # Обучаем каждый экстрактор
        self.text_extractor.fit(data)
        self.image_extractor.fit(data)
        self.struct_extractor.fit(data)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение всех признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с извлеченными признаками
        """
        self.logger.info("Извлечение всех признаков")

        # Извлекаем признаки для каждого типа
        text_features = self.text_extractor.transform(data)
        image_features = self.image_extractor.transform(data)
        struct_features = self.struct_extractor.transform(data)

        # Объединяем все признаки
        all_features = pd.concat([text_features, image_features, struct_features], axis=1)

        # Снижаем размерность для эмбеддингов
        reduced_features = self._reduce_dimensions(all_features)

        self.logger.info(f"Извлечено всего {reduced_features.shape[1]} признаков")
        return reduced_features