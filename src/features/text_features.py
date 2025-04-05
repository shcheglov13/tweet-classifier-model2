import pandas as pd
import numpy as np
import re
import emoji
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

from src.features.feature_base import FeatureExtractorBase
from src.utils.caching import cache_feature
from src.utils.progress_tracker import ProgressTracker


class TextFeatureExtractor(FeatureExtractorBase):
    """
    Класс для извлечения текстовых признаков из твитов.
    """

    def __init__(
            self,
            bert_model_name: str = 'vinai/bertweet-base',
            cache_dir: Optional[Union[str, Path]] = None,
            batch_size: int = 32,
            device: Optional[str] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация экстрактора текстовых признаков.

        Args:
            bert_model_name: Название модели BERT для извлечения эмбеддингов
            cache_dir: Директория для кеширования
            batch_size: Размер батча для обработки
            device: Устройство для вычислений ('cpu' или 'cuda')
            logger: Объект логгера
        """
        self.bert_model_name = bert_model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)

        self.tokenizer = None
        self.model = None

    def _load_bert_model(self):
        """Загрузка модели BERT и токенизатора."""
        if self.tokenizer is None or self.model is None:
            self.logger.info(f"Загрузка модели BERTweet: {self.bert_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.model = AutoModel.from_pretrained(self.bert_model_name)
            self.model.to(self.device)
            self.model.eval()

    def fit(self, data: pd.DataFrame) -> 'TextFeatureExtractor':
        """
        Подготовка экстрактора (загрузка моделей и т.д.).

        Args:
            data: Датафрейм с данными

        Returns:
            self: Возвращает экземпляр класса
        """
        self._load_bert_model()
        return self

    @cache_feature(feature_type='text_embeddings')
    def _extract_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Извлечение эмбеддингов из текстов с помощью BERTweet.

        Args:
            texts: Список текстов

        Returns:
            np.ndarray: Матрица эмбеддингов
        """
        self._load_bert_model()

        embeddings = []

        # Создаем "пустой" эмбеддинг для отсутствующих текстов
        empty_embedding = np.zeros(self.model.config.hidden_size)

        with ProgressTracker(
                total=len(texts),
                description="Извлечение BERT эмбеддингов",
                logger=self.logger
        ) as progress:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Проверяем, какие тексты пустые или None
                is_empty = [text is None or text == "" or pd.isna(text) for text in batch_texts]

                # Если все тексты в батче пустые
                if all(is_empty):
                    embeddings.extend([empty_embedding] * len(batch_texts))
                    progress.update(len(batch_texts))
                    continue

                # Заменяем None и пустые строки на пробел (минимальный непустой текст)
                processed_texts = [text if not (text is None or text == "" or pd.isna(text)) else " " for text in
                                   batch_texts]

                encoded_input = self.tokenizer(
                    processed_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # Используем [CLS] токен в качестве представления всего предложения
                batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

                # Заменяем эмбеддинги для пустых текстов на нулевые
                for j, empty in enumerate(is_empty):
                    if empty:
                        batch_embeddings[j] = empty_embedding

                embeddings.append(batch_embeddings)

                # Обновляем прогресс
                progress.update(len(batch_texts))

        return np.vstack(embeddings) if embeddings else np.array([])

    def _count_special_elements(self, text: str) -> Dict[str, int]:
        """
        Подсчет специальных элементов в тексте.

        Args:
            text: Текст для анализа

        Returns:
            Dict[str, int]: Словарь с количеством разных элементов
        """
        if text is None or pd.isna(text):
            return {
                'hashtag_count': 0,
                'mention_count': 0,
                'url_count': 0,
                'emoji_count': 0
            }

        hashtags = len(re.findall(r'#\w+', text))
        mentions = len(re.findall(r'@\w+', text))
        urls = len(re.findall(r'https?://\S+', text))
        emoji_count = len([c for c in text if c in emoji.EMOJI_DATA])

        return {
            'hashtag_count': hashtags,
            'mention_count': mentions,
            'url_count': urls,
            'emoji_count': emoji_count
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение текстовых признаков.

        Args:
            data: Датафрейм с данными

        Returns:
            pd.DataFrame: Датафрейм с извлеченными признаками
        """
        self.logger.info("Извлечение текстовых признаков")

        features = {}

        # Длина текста
        features['text_length'] = data['text'].apply(lambda x: len(x) if pd.notna(x) else 0)
        features['quoted_text_length'] = data['quoted_text'].apply(lambda x: len(x) if pd.notna(x) else 0)
        features['combined_text_length'] = features['text_length'] + features['quoted_text_length']

        # Количество слов
        features['text_word_count'] = data['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        features['quoted_text_word_count'] = data['quoted_text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0)
        features['combined_word_count'] = features['text_word_count'] + features['quoted_text_word_count']

        # Средняя длина слов
        features['avg_word_length_text'] = data['text'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and str(x).split() else 0
        )
        features['avg_word_length_quoted'] = data['quoted_text'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and str(x).split() else 0
        )

        # Специальные элементы
        special_elements = data['text'].apply(self._count_special_elements)
        for key in ['hashtag_count', 'mention_count', 'url_count', 'emoji_count']:
            features[key] = special_elements.apply(lambda x: x[key])

        # Плотность специальных элементов
        for element in ['hashtag', 'mention', 'url', 'emoji']:
            count_col = f'{element}_count'
            density_col = f'{element}_density'
            # Заменяем 0 на 1 для предотвращения деления на ноль
            features[density_col] = features[count_col] / features['text_length'].replace(0, 1)

        # Доля заглавных букв
        features['uppercase_ratio'] = data['text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x).replace(" ", "")), 1) if pd.notna(x) else 0
        )

        # Удлиненные слова и избыточная пунктуация
        features['word_elongation_count'] = data['text'].apply(
            lambda x: len(re.findall(r'\b\w*(\w)\1{2,}\w*\b', str(x))) if pd.notna(x) else 0
        )
        features['excessive_punctuation_count'] = data['text'].apply(
            lambda x: len(re.findall(r'[!?\.]{2,}', str(x))) if pd.notna(x) else 0
        )

        # Извлечение BERT эмбеддингов
        text_embeddings = self._extract_bert_embeddings(data['text'].tolist())
        quoted_text_embeddings = self._extract_bert_embeddings(data['quoted_text'].tolist())

        # Добавляем эмбеддинги в словарь
        for i in range(text_embeddings.shape[1]):
            features[f'text_emb_{i}'] = text_embeddings[:, i]

        for i in range(quoted_text_embeddings.shape[1]):
            features[f'quoted_text_emb_{i}'] = quoted_text_embeddings[:, i]

        features_df = pd.DataFrame(features, index=data.index)
        self.logger.info(f"Извлечено {features_df.shape[1]} текстовых признаков")
        return features_df
