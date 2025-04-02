import pandas as pd
import numpy as np
import logging
import time
import mlflow
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
from flaml import AutoML
from src.utils.gpu_utils import get_gpu_settings_for_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


class ModelTrainer:
    """
    Класс для обучения моделей с помощью FLAML.
    """

    def __init__(
            self,
            time_budget: int = 3600,  # 1 час
            estimator_list: Optional[List[str]] = None,
            metric: str = 'average_precision',
            task: str = 'classification',
            n_folds: int = 5,
            ensemble_type: str = 'stack',
            use_gpu: bool = True,
            output_dir: Union[str, Path] = 'models',
            random_state: int = 42,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация тренера моделей.

        Args:
            time_budget: Бюджет времени на обучение (в секундах)
            estimator_list: Список используемых моделей
            metric: Метрика для оптимизации
            task: Тип задачи ('classification' или 'regression')
            n_folds: Количество фолдов для кросс-валидации
            ensemble_type: Тип ансамбля ('stack' или 'best')
            use_gpu: Использовать ли GPU для обучения
            output_dir: Директория для сохранения моделей
            random_state: Состояние генератора случайных чисел
            logger: Объект логгера
        """
        self.time_budget = time_budget

        if estimator_list is None:
            self.estimator_list = ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree', 'mlp']
        else:
            self.estimator_list = estimator_list

        self.metric = metric
        self.task = task
        self.n_folds = n_folds
        self.ensemble_type = ensemble_type
        self.use_gpu = use_gpu
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.automl = None
        self.feature_importance = None
        self.metrics = {}

        # Валидируем список моделей
        valid_estimators = ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree', 'mlp']
        for estimator in self.estimator_list:
            if estimator not in valid_estimators:
                self.logger.warning(f"Неизвестный тип модели: {estimator}. Будет пропущен.")

        self.estimator_list = [est for est in self.estimator_list if est in valid_estimators]
        if not self.estimator_list:
            self.logger.warning("Не указаны валидные модели. Будут использованы все доступные.")
            self.estimator_list = valid_estimators

    def _prepare_gpu_settings(self) -> Dict:
        """
        Подготовка настроек для обучения на GPU.

        Returns:
            Dict: Настройки для моделей
        """
        if not self.use_gpu:
            return {}

        # Настройки для разных моделей
        gpu_settings = {}
        for model_type in self.estimator_list:
            settings = get_gpu_settings_for_model(model_type)
            if settings:
                gpu_settings[model_type] = settings

        return gpu_settings

    def _calculate_class_weights(self, y: pd.Series) -> Dict:
        """
        Расчет весов классов для несбалансированных данных.

        Args:
            y: Серия с метками классов

        Returns:
            Dict: Словарь с весами классов
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Рассчитываем веса обратно пропорциональные частоте классов
        class_counts = y.value_counts()
        weights = {cls: n_samples / (n_classes * count) for cls, count in class_counts.items()}

        self.logger.info(f"Рассчитаны веса классов: {weights}")
        return weights

    def train(self, X: pd.DataFrame, y: pd.Series) -> 'ModelTrainer':
        """
        Обучение модели с использованием FLAML.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной

        Returns:
            self: Возвращает экземпляр класса
        """
        self.logger.info(f"Начало обучения модели с бюджетом времени {self.time_budget} секунд")
        self.logger.info(f"Используемые модели: {', '.join(self.estimator_list)}")

        # Инициализация AutoML
        self.automl = AutoML()

        # Подготовка настроек
        gpu_settings = self._prepare_gpu_settings()
        class_weights = self._calculate_class_weights(y)

        # Логирование процесса обучения
        mlflow.log_param("estimator_list", self.estimator_list)
        mlflow.log_param("metric", self.metric)
        mlflow.log_param("ensemble_type", self.ensemble_type)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("class_distribution", dict(y.value_counts()))

        # Засекаем время
        start_time = time.time()

        # Обучение модели
        self.automl.fit(
            X=X.values,
            y=y.values,
            task=self.task,
            time_budget=self.time_budget,
            metric=self.metric,
            estimator_list=self.estimator_list,
            ensemble=True,
            ensemble_type=self.ensemble_type,
            eval_method='cv',
            n_splits=self.n_folds,
            custom_hp=gpu_settings,
            verbose=1,
            seed=self.random_state,
            class_weight=class_weights
        )

        # Время обучения
        training_time = time.time() - start_time
        self.logger.info(f"Обучение завершено за {training_time:.2f} секунд")

        # Логируем результаты
        self.logger.info(f"Лучшая модель: {self.automl.best_estimator}")
        self.logger.info(f"Лучшая конфигурация: {self.automl.best_config}")
        self.logger.info(f"Лучшее значение метрики: {1 - self.automl.best_loss:.4f}")

        mlflow.log_param("best_estimator", self.automl.best_estimator)
        mlflow.log_param("best_config", self.automl.best_config)
        mlflow.log_metric("best_score", 1 - self.automl.best_loss)
        mlflow.log_metric("training_time", training_time)

        # Извлекаем важность признаков, если она доступна
        try:
            if hasattr(self.automl.model.estimator, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': self.automl.model.estimator.feature_importances_
                }).sort_values('Importance', ascending=False)

                mlflow.log_param("top_features", self.feature_importance['Feature'].head(10).tolist())
        except Exception as e:
            self.logger.warning(f"Не удалось извлечь важность признаков: {e}")

        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Оценка модели на тестовых данных.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной

        Returns:
            Dict: Словарь с метриками качества
        """
        if self.automl is None:
            self.logger.error("Модель не обучена")
            return {}

        self.logger.info("Оценка модели на тестовых данных")

        # Получаем предсказания
        y_pred_proba = self.automl.predict_proba(X.values)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Рассчитываем метрики
        self.metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'pr_auc': average_precision_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }

        # Логируем метрики
        for metric_name, metric_value in self.metrics.items():
            if metric_name != 'confusion_matrix':
                mlflow.log_metric(metric_name, metric_value)
                self.logger.info(f"{metric_name}: {metric_value:.4f}")

        return self.metrics

    def save_model(self, filename: str = 'model.joblib') -> Path:
        """
        Сохранение модели на диск.

        Args:
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненной модели
        """
        if self.automl is None:
            self.logger.error("Модель не обучена")
            return None

        model_path = self.output_dir / filename
        self.logger.info(f"Сохранение модели в {model_path}")

        joblib.dump(self.automl, model_path)
        mlflow.log_artifact(str(model_path))

        return model_path

    def load_model(self, model_path: Union[str, Path]) -> 'ModelTrainer':
        """
        Загрузка сохраненной модели.

        Args:
            model_path: Путь к сохраненной модели

        Returns:
            self: Возвращает экземпляр класса
        """
        model_path = Path(model_path)
        if not model_path.exists():
            self.logger.error(f"Файл модели не найден: {model_path}")
            return self

        self.logger.info(f"Загрузка модели из {model_path}")
        self.automl = joblib.load(model_path)

        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Получение важности признаков.

        Returns:
            pd.DataFrame: Датафрейм с важностью признаков
        """
        if self.feature_importance is None:
            self.logger.warning("Информация о важности признаков недоступна")
            return pd.DataFrame()

        return self.feature_importance