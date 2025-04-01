# src/models/model_config.py
from typing import Dict, Any, List, Optional, Union
import logging


class ModelConfig:
    """
    Класс для управления конфигурациями моделей машинного обучения.
    """

    # Базовые настройки для различных моделей
    DEFAULT_CONFIGS = {
        'lgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        },
        'xgboost': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 42
        },
        'catboost': {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': 42,
            'verbose': 0
        },
        'rf': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42
        },
        'extra_tree': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42
        },
        'mlp': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'shuffle': True,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
    }

    # GPU-настройки для моделей
    GPU_CONFIGS = {
        'lgbm': {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        },
        'xgboost': {
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        },
        'catboost': {
            'task_type': 'GPU',
            'devices': '0'
        }
    }

    def __init__(
            self,
            model_type: str,
            use_gpu: bool = True,
            custom_config: Optional[Dict[str, Any]] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация конфигурации модели.

        Args:
            model_type: Тип модели ('lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree', 'mlp')
            use_gpu: Использовать ли GPU (если доступен)
            custom_config: Пользовательская конфигурация (переопределяет базовую)
            logger: Объект логгера
        """
        self.model_type = model_type.lower()
        self.use_gpu = use_gpu
        self.custom_config = custom_config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Проверяем, поддерживается ли указанный тип модели
        if self.model_type not in self.DEFAULT_CONFIGS:
            raise ValueError(
                f"Неподдерживаемый тип модели: {model_type}. Допустимые типы: {list(self.DEFAULT_CONFIGS.keys())}")

        # Создаем базовую конфигурацию
        self.config = self._create_config()

    def _create_config(self) -> Dict[str, Any]:
        """
        Создание конфигурации модели с учетом базовых настроек, GPU и пользовательских настроек.

        Returns:
            Dict[str, Any]: Итоговая конфигурация
        """
        # Начинаем с базовой конфигурации
        config = self.DEFAULT_CONFIGS[self.model_type].copy()

        # Добавляем настройки GPU, если нужно
        if self.use_gpu and self.model_type in self.GPU_CONFIGS:
            gpu_config = self.GPU_CONFIGS[self.model_type].copy()

            # Проверяем доступность GPU
            if self.model_type in ['lgbm', 'xgboost', 'catboost']:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.logger.warning("GPU не доступен. Используем CPU.")
                        return config
                except ImportError:
                    self.logger.warning("Не удалось проверить доступность GPU. Используем CPU.")
                    return config

            config.update(gpu_config)
            self.logger.info(f"Добавлены настройки GPU для {self.model_type}")

        # Добавляем пользовательские настройки (они имеют наивысший приоритет)
        if self.custom_config:
            config.update(self.custom_config)

        return config

    def get_config(self) -> Dict[str, Any]:
        """
        Получение текущей конфигурации модели.

        Returns:
            Dict[str, Any]: Конфигурация модели
        """
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Обновление конфигурации модели.

        Args:
            updates: Словарь с обновлениями для конфигурации
        """
        self.config.update(updates)

    def get_flaml_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Получение конфигурации для FLAML.

        Returns:
            Dict[str, Dict[str, Any]]: Конфигурация для FLAML
        """
        return {self.model_type: self.config}

    @classmethod
    def get_ensemble_config(
            cls,
            estimator_list: List[str],
            use_gpu: bool = True,
            ensemble_type: str = 'stack',
            custom_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Получение конфигурации для ансамбля моделей.

        Args:
            estimator_list: Список типов моделей для ансамбля
            use_gpu: Использовать ли GPU (если доступен)
            ensemble_type: Тип ансамбля ('stack' или 'best')
            custom_configs: Пользовательские конфигурации для моделей

        Returns:
            Dict[str, Any]: Конфигурация для ансамбля моделей
        """
        # Проверяем, что все типы моделей поддерживаются
        for model_type in estimator_list:
            if model_type not in cls.DEFAULT_CONFIGS:
                raise ValueError(
                    f"Неподдерживаемый тип модели: {model_type}. Допустимые типы: {list(cls.DEFAULT_CONFIGS.keys())}")

        # Создаем конфигурации для каждой модели
        configs = {}
        for model_type in estimator_list:
            custom_config = custom_configs.get(model_type, {}) if custom_configs else {}
            model_config = cls(model_type, use_gpu, custom_config)
            configs[model_type] = model_config.get_config()

        # Добавляем настройки ансамбля
        ensemble_config = {
            'ensemble': True,
            'ensemble_type': ensemble_type
        }

        return {'custom_hp': configs, **ensemble_config}

    @staticmethod
    def get_model_metrics(model_type: str) -> List[str]:
        """
        Получение списка метрик, поддерживаемых моделью.

        Args:
            model_type: Тип модели

        Returns:
            List[str]: Список поддерживаемых метрик
        """
        model_type = model_type.lower()

        if model_type == 'lgbm':
            return ['binary_logloss', 'binary_error', 'auc', 'average_precision']
        elif model_type == 'xgboost':
            return ['logloss', 'error', 'auc', 'aucpr']
        elif model_type == 'catboost':
            return ['Logloss', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
        elif model_type in ['rf', 'extra_tree', 'mlp']:
            return ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
        else:
            return []

    @staticmethod
    def map_metric_to_flaml(metric: str) -> str:
        """
        Преобразование названия метрики в формат, понятный FLAML.

        Args:
            metric: Название метрики

        Returns:
            str: Название метрики для FLAML
        """
        metric_map = {
            'accuracy': 'accuracy',
            'error': 'error',
            'logloss': 'log_loss',
            'binary_logloss': 'log_loss',
            'Logloss': 'log_loss',
            'auc': 'roc_auc',
            'roc_auc': 'roc_auc',
            'AUC': 'roc_auc',
            'aucpr': 'average_precision',
            'average_precision': 'average_precision',
            'f1': 'f1',
            'F1': 'f1',
            'precision': 'precision',
            'Precision': 'precision',
            'recall': 'recall',
            'Recall': 'recall',
            'mse': 'mse',
            'rmse': 'rmse',
            'mae': 'mae'
        }

        return metric_map.get(metric, metric)