import logging
import torch
import numpy as np
from typing import Dict, Any, Optional


def check_gpu_availability() -> bool:
    """
    Проверка доступности GPU с CUDA.

    Returns:
        bool: True, если GPU доступен, иначе False
    """
    return torch.cuda.is_available()


def get_gpu_info() -> Dict[str, Any]:
    """
    Получение информации о доступных GPU.

    Returns:
        Dict[str, Any]: Словарь с информацией о GPU
    """
    if not check_gpu_availability():
        return {'available': False, 'count': 0, 'devices': []}

    gpu_count = torch.cuda.device_count()
    devices = []

    for i in range(gpu_count):
        device_info = {
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),  # В ГБ
            'memory_allocated': torch.cuda.memory_allocated(i) / (1024 ** 3),  # В ГБ
            'memory_cached': torch.cuda.memory_reserved(i) / (1024 ** 3)  # В ГБ
        }
        devices.append(device_info)

    return {
        'available': True,
        'count': gpu_count,
        'devices': devices,
        'current_device': torch.cuda.current_device()
    }


def get_optimal_device() -> torch.device:
    """
    Получение оптимального устройства для вычислений (GPU или CPU).

    Returns:
        torch.device: Оптимальное устройство
    """
    if check_gpu_availability():
        # Выбираем GPU с наибольшим свободным объемом памяти
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            free_memory = []

            for i in range(gpu_count):
                total = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                free = total - allocated
                free_memory.append((i, free))

            # Сортируем по убыванию свободной памяти
            free_memory.sort(key=lambda x: x[1], reverse=True)
            best_gpu = free_memory[0][0]

            return torch.device(f"cuda:{best_gpu}")
        else:
            return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_gpu_settings_for_model(model_type: str, gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Получение настроек GPU для конкретной модели.

    Args:
        model_type: Тип модели ('lgbm', 'xgboost', 'catboost', etc.)
        gpu_id: ID конкретного GPU для использования

    Returns:
        Dict[str, Any]: Словарь с настройками GPU для модели
    """
    if not check_gpu_availability():
        return {}

    # Если GPU ID не указан, выбираем оптимальный
    if gpu_id is None:
        device = get_optimal_device()
        if device.type == 'cuda':
            gpu_id = device.index
        else:
            return {}  # Нет доступных GPU

    # Настройки для разных типов моделей
    if model_type.lower() == 'lgbm':
        return {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': gpu_id}

    elif model_type.lower() == 'xgboost':
        return {'tree_method': 'gpu_hist', 'gpu_id': gpu_id}

    elif model_type.lower() == 'catboost':
        return {'task_type': 'GPU', 'devices': str(gpu_id)}

    elif model_type.lower() == 'pytorch':
        return {'device': f'cuda:{gpu_id}'}

    else:
        return {}


def set_memory_growth(enable: bool = True) -> None:
    """
    Настройка динамического выделения памяти GPU для TensorFlow.

    Args:
        enable: Включить ли динамическое выделение памяти
    """
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable)
    except (ImportError, AttributeError, RuntimeError) as e:
        logging.warning(f"Не удалось настроить динамическое выделение памяти: {e}")


def get_gpu_batch_size(
        base_batch_size: int,
        available_memory_gb: Optional[float] = None,
        memory_per_sample_mb: float = 50.0
) -> int:
    """
    Расчет оптимального размера батча в зависимости от доступной памяти GPU.

    Args:
        base_batch_size: Базовый размер батча для CPU
        available_memory_gb: Доступная память GPU в ГБ (если None, определяется автоматически)
        memory_per_sample_mb: Приблизительное потребление памяти на один образец в МБ

    Returns:
        int: Оптимальный размер батча
    """
    if not check_gpu_availability():
        return base_batch_size

    # Определяем доступную память, если не указана
    if available_memory_gb is None:
        device = get_optimal_device()
        if device.type == 'cuda':
            gpu_id = device.index
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated = torch.cuda.memory_allocated(gpu_id)
            free = (total - allocated) / (1024 ** 3)  # В ГБ

            # Оставляем резерв 20%
            available_memory_gb = free * 0.8
        else:
            return base_batch_size

    # Рассчитываем размер батча
    max_samples = int((available_memory_gb * 1024) / memory_per_sample_mb)

    # Округляем до ближайшей степени 2 для оптимальной производительности
    power_of_2 = 2 ** int(np.log2(max_samples))

    # Не меньше базового размера и не больше рассчитанного максимума
    return max(base_batch_size, min(max_samples, power_of_2))


def clean_gpu_memory() -> None:
    """
    Очистка памяти GPU.
    """
    if check_gpu_availability():
        torch.cuda.empty_cache()