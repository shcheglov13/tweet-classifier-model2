import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image
import io
import logging
from typing import List, Optional, Union
from pathlib import Path

from src.features.feature_base import FeatureExtractorBase
from src.utils.caching import cache_feature, cache_image
from src.utils.progress_tracker import ProgressTracker


class ImageFeatureExtractor(FeatureExtractorBase):
    """
    Класс для извлечения признаков из изображений твитов.
    """

    def __init__(
            self,
            clip_model_name: str = 'openai/clip-vit-large-patch14',
            cache_dir: Optional[Union[str, Path]] = None,
            batch_size: int = 4,
            device: Optional[str] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация экстрактора визуальных признаков.

        Args:
            clip_model_name: Название модели CLIP для извлечения эмбеддингов
            cache_dir: Директория для кеширования
            batch_size: Размер батча для обработки
            device: Устройство для вычислений ('cpu' или 'cuda')
            logger: Объект логгера
        """
        self.clip_model_name = clip_model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)

        self.processor = None
        self.model = None

    def _load_clip_model(self):
        """Загрузка модели CLIP и процессора."""
        if self.processor is None or self.model is None:
            self.logger.info(f"Загрузка модели CLIP: {self.clip_model_name}")
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name, use_fast=True)
            self.model = CLIPModel.from_pretrained(self.clip_model_name)
            self.model.to(self.device)
            self.model.eval()

    @cache_image
    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Загрузка изображения по URL.

        Args:
            image_url: URL изображения

        Returns:
            Optional[Image.Image]: Загруженное изображение или None в случае ошибки
        """
        if pd.isna(image_url) or not image_url:
            return None

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        except Exception as e:
            self.logger.warning(f"Ошибка загрузки изображения {image_url}: {e}")
            return None

    @cache_feature(feature_type='clip_embeddings')
    def _extract_clip_embeddings(self, image_urls: List[str]) -> np.ndarray:
        """
        Извлечение эмбеддингов из изображений с помощью CLIP.

        Args:
            image_urls: Список URL изображений

        Returns:
            np.ndarray: Матрица эмбеддингов
        """
        self._load_clip_model()

        embeddings = []

        # Создаем "пустой" эмбеддинг для отсутствующих изображений
        empty_embedding = np.zeros(self.model.config.projection_dim)

        with ProgressTracker(
                total=len(image_urls),
                description="Извлечение CLIP эмбеддингов",
                logger=self.logger
        ) as progress:
            for i in range(0, len(image_urls), self.batch_size):
                batch_urls = image_urls[i:i + self.batch_size]

                # Загружаем изображения (с использованием кеша)
                batch_images = [self._download_image(url) for url in batch_urls]

                # Пропускаем батчи без изображений
                if all(img is None for img in batch_images):
                    embeddings.extend([empty_embedding] * len(batch_images))
                    progress.update(len(batch_images))
                    continue

                # Заменяем отсутствующие изображения на пустые изображения для батча
                valid_images = []
                valid_indices = []

                for j, img in enumerate(batch_images):
                    if img is not None:
                        valid_images.append(img)
                        valid_indices.append(j)

                if not valid_images:
                    embeddings.extend([empty_embedding] * len(batch_images))
                    progress.update(len(batch_images))
                    continue

                # Обрабатываем и получаем эмбеддинги для валидных изображений
                try:
                    inputs = self.processor(images=valid_images, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)

                    batch_embeddings = outputs.cpu().numpy()

                    # Распределяем эмбеддинги по исходным индексам
                    batch_result = [empty_embedding] * len(batch_images)
                    for embed_idx, orig_idx in enumerate(valid_indices):
                        batch_result[orig_idx] = batch_embeddings[embed_idx]

                    embeddings.extend(batch_result)

                except Exception as e:
                    self.logger.error(f"Ошибка при извлечении CLIP эмбеддингов: {e}")
                    embeddings.extend([empty_embedding] * len(batch_images))

                # Обновляем прогресс
                progress.update(len(batch_images))

                # Добавляем информацию о статистике обработки
                valid_count = len(valid_images)
                progress.set_postfix(valid_images=valid_count, invalid_images=len(batch_images) - valid_count)

        return np.vstack(embeddings)

    def fit(self, data: pd.DataFrame) -> 'ImageFeatureExtractor':
        """
        Подготовка экстрактора.

        Args:
            data: Датафрейм с данными

        Returns:
            self: Возвращает экземпляр класса
        """
        self._load_clip_model()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение визуальных признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с извлеченными признаками
        """
        self.logger.info("Извлечение визуальных признаков")

        # Создаем пустой датафрейм для признаков
        features_df = pd.DataFrame(index=data.index)

        # Извлечение CLIP эмбеддингов
        self.logger.info("Извлечение CLIP эмбеддингов для изображений")
        clip_embeddings = self._extract_clip_embeddings(data['image_url'].tolist())

        emb_columns = [f'image_emb_{i}' for i in range(clip_embeddings.shape[1])]
        emb_df = pd.DataFrame(clip_embeddings, index=data.index, columns=emb_columns)

        # Объединяем все данные в один DataFrame
        features_df = pd.concat([features_df, emb_df], axis=1)

        self.logger.info(f"Извлечено {features_df.shape[1]} визуальных признаков")
        return features_df