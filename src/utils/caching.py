import json
import pickle
import hashlib
import functools
from pathlib import Path
from typing import Any, Callable, Union
import numpy as np
import pandas as pd
import logging


def _create_cache_dir(cache_dir: Union[str, Path]) -> Path:
    """
    Создание директории для кеширования.

    Args:
        cache_dir: Путь к директории для кеширования

    Returns:
        Path: Путь к созданной директории
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _get_hash(obj: Any) -> str:
    """
    Получение хеша объекта для использования в кеше.

    Args:
        obj: Объект для хеширования

    Returns:
        str: Хеш объекта
    """
    if isinstance(obj, (str, int, float, bool)):
        data = str(obj).encode('utf-8')
    elif isinstance(obj, list):
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
    elif isinstance(obj, dict):
        data = json.dumps({str(k): str(v) for k, v in obj.items()}, sort_keys=True).encode('utf-8')
    elif isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        data = pickle.dumps(obj)
    else:
        data = str(obj).encode('utf-8')

    return hashlib.md5(data).hexdigest()


def cache_feature(feature_type: str = 'default'):
    """
    Декоратор для кеширования результатов извлечения признаков.

    Args:
        feature_type: Тип признака для организации кеша

    Returns:
        Callable: Декорированная функция
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Если кеширование отключено, просто вызываем функцию
            if not hasattr(self, 'cache_dir') or self.cache_dir is None:
                return func(self, *args, **kwargs)

            # Создаем директорию для кеша
            cache_dir = _create_cache_dir(self.cache_dir / feature_type)

            # Создаем уникальный хеш для аргументов
            args_hash = _get_hash(args)
            kwargs_hash = _get_hash(kwargs)
            cache_file = cache_dir / f"{func.__name__}_{args_hash}_{kwargs_hash}.pkl"

            # Если кеш существует, загружаем его
            if cache_file.exists():
                try:
                    logger = getattr(self, 'logger', logging.getLogger(__name__))
                    logger.info(f"Загрузка кешированных признаков из {cache_file}")
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger = getattr(self, 'logger', logging.getLogger(__name__))
                    logger.warning(f"Ошибка загрузки кеша: {e}. Пересчитываем признаки.")

            # Если кеша нет или произошла ошибка, вычисляем признаки
            result = func(self, *args, **kwargs)

            # Сохраняем результат в кеш
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.info(f"Признаки сохранены в кеш: {cache_file}")
            except Exception as e:
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.warning(f"Ошибка сохранения в кеш: {e}")

            return result
        return wrapper
    return decorator


def cache_image(func: Callable):
    """
    Декоратор для кеширования изображений.

    Args:
        func: Функция для декорирования

    Returns:
        Callable: Декорированная функция
    """
    @functools.wraps(func)
    def wrapper(self, image_url: str, *args, **kwargs):
        # Если кеширование отключено, просто вызываем функцию
        if not hasattr(self, 'cache_dir') or self.cache_dir is None:
            return func(self, image_url, *args, **kwargs)

        # Если URL пустой, просто вызываем функцию
        if image_url is None or pd.isna(image_url) or image_url == "":
            return func(self, image_url, *args, **kwargs)

        # Создаем директорию для кеша изображений
        cache_dir = _create_cache_dir(self.cache_dir / 'images')

        # Создаем уникальный хеш для URL
        url_hash = _get_hash(image_url)
        cache_file = cache_dir / f"{url_hash}.pkl"

        # Если кеш существует, загружаем его
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.warning(f"Ошибка загрузки кеша изображения: {e}. Загружаем заново.")

        # Если кеша нет или произошла ошибка, загружаем изображение
        image = func(self, image_url, *args, **kwargs)

        # Сохраняем изображение в кеш
        if image is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(image, f)
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.info(f"Изображение сохранено в кеш: {cache_file}")
            except Exception as e:
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.warning(f"Ошибка сохранения изображения в кеш: {e}")

        return image
    return wrapper