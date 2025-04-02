import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Union
from pathlib import Path

from src.features.feature_base import FeatureExtractorBase


class StructuralFeatureExtractor(FeatureExtractorBase):
    """
    Класс для извлечения структурных признаков из твитов.
    """

    def __init__(
            self,
            cache_dir: Optional[Union[str, Path]] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация экстрактора структурных признаков.

        Args:
            cache_dir: Директория для кеширования
            logger: Объект логгера
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, data: pd.DataFrame) -> 'StructuralFeatureExtractor':
        """
        Подготовка экстрактора.

        Args:
            data: Датафрейм с данными

        Returns:
            self: Возвращает экземпляр класса
        """
        
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение структурных признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с извлеченными признаками
        """
        self.logger.info("Извлечение структурных признаков")

        # Создаем пустой датафрейм для признаков
        features_df = pd.DataFrame(index=data.index)

        # One-hot кодирование типа твита
        tweet_types = pd.get_dummies(data['tweet_type'], prefix='tweet_type')
        features_df = pd.concat([features_df, tweet_types], axis=1)

        # Флаги наличия контента
        features_df['has_main_text'] = (~data['text'].isna() & (data['text'] != "")).astype(int)
        features_df['has_quoted_text'] = (~data['quoted_text'].isna() & (data['quoted_text'] != "")).astype(int)
        features_df['has_image'] = (~data['image_url'].isna() & (data['image_url'] != "")).astype(int)

        # Тип медиа
        features_df['media_type_only_text'] = (
                    (features_df['has_main_text'] | features_df['has_quoted_text']) & ~features_df['has_image']).astype(
            int)
        features_df['media_type_image'] = features_df['has_image'].astype(int)
        features_df['media_type_video'] = data['image_url'].apply(self._is_video_url).astype(int)

        # Соотношение длин текстов
        text_lengths = data['text'].fillna('').apply(len)
        quoted_text_lengths = data['quoted_text'].fillna('').apply(len)

        features_df['text_quoted_ratio'] = 0.0
        mask = (quoted_text_lengths > 0)
        if mask.any():
            features_df.loc[mask, 'text_quoted_ratio'] = text_lengths[mask] / quoted_text_lengths[mask]

        # Временные признаки из даты публикации
        def extract_time_features(date_str):
            if pd.isna(date_str):
                return pd.Series({'hour': 0, 'day_of_week': 0, 'is_weekend': 0})

            try:
                date_obj = datetime.fromisoformat(str(date_str).split('+')[0].strip())
                hour = date_obj.hour
                day_of_week = date_obj.weekday()  # 0-6, где 0 - понедельник
                is_weekend = 1 if day_of_week >= 5 else 0  # 5, 6 - выходные (суббота, воскресенье)

                return pd.Series({'hour': hour, 'day_of_week': day_of_week, 'is_weekend': is_weekend})
            except Exception as e:
                self.logger.warning(f"Ошибка при извлечении временных признаков: {e}")
                return pd.Series({'hour': 0, 'day_of_week': 0, 'is_weekend': 0})

        time_features = data['created_at'].apply(extract_time_features)
        features_df = pd.concat([features_df, time_features], axis=1)

        self.logger.info(f"Извлечено {features_df.shape[1]} структурных признаков")
        return features_df

    def _is_video_url(self, url: str) -> bool:
        """
        Проверка, является ли URL ссылкой на видео.

        Args:
            url: URL для проверки

        Returns:
            bool: True, если URL указывает на видео, иначе False
        """
        if url is None or pd.isna(url) or not url:
            return False

        video_patterns = [
            'ext_tw_video_thumb',
            'amplify_video_thumb',
            'tweet_video_thumb'
        ]

        return any(pattern in url for pattern in video_patterns)