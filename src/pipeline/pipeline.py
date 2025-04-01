import pandas as pd
import numpy as np
import mlflow
import time
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.data_splitter import DataSplitter
from src.features.feature_extractor import FeatureExtractor
from src.preprocessing.missing_values import MissingValuesHandler
from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.target_binarizer import TargetBinarizer
from src.feature_selection.correlation import CorrelationSelector
from src.feature_selection.boruta import BorutaSelector
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.visualization import ModelVisualizer
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.mlflow_utils import setup_mlflow, log_config, log_model_metrics
from src.utils.gpu_utils import check_gpu_availability, get_gpu_info


class Pipeline:
    """
    Основной класс для координации всего пайплайна обучения модели.
    """

    def __init__(
            self,
            config_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            experiment_name: str = 'tweet_classification'
    ):
        """
        Инициализация пайплайна.

        Args:
            config_path: Путь к файлу конфигурации
            output_dir: Директория для сохранения результатов
            experiment_name: Название эксперимента в MLflow
        """
        # Загрузка конфигурации
        self.config_path = Path(config_path)
        self.config = Config(config_path)

        # Инициализация директорий
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_dir is None:
            output_dir = self.config.get('output_dir', 'outputs')

        self.output_dir = Path(output_dir) / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настройка логирования
        log_config = self.config.get('logging_config')
        self.logger = setup_logger(
            log_file=self.output_dir / 'pipeline.log',
            level=self.config.get('logging_level', 'INFO'),
            config_file=log_config if log_config else None
        )

        # Проверка доступности GPU
        self.use_gpu = self.config.get('use_gpu', True)
        if self.use_gpu:
            gpu_available = check_gpu_availability()
            if gpu_available:
                gpu_info = get_gpu_info()
                self.logger.info(f"Доступно {gpu_info['count']} GPU устройств")
                for i, device in enumerate(gpu_info['devices']):
                    self.logger.info(f"GPU {i}: {device['name']}, {device['memory_total']:.2f} GB")
            else:
                self.logger.warning("GPU недоступен. Будет использован CPU.")
                self.use_gpu = False

        # Настройка MLflow
        self.mlflow_client = setup_mlflow(
            experiment_name=experiment_name,
            tracking_uri=self.config.get('mlflow_tracking_uri', 'mlruns')
        )

        self.logger.info(f"Инициализирован пайплайн с конфигурацией из {config_path}")
        self.logger.info(f"Результаты будут сохранены в {self.output_dir}")

    def run(self) -> Dict:
        """
        Запуск полного пайплайна обучения модели.

        Returns:
            Dict: Словарь с результатами выполнения
        """
        start_time = time.time()
        self.logger.info("Начало выполнения пайплайна")

        with mlflow.start_run() as run:
            try:
                # Логируем конфигурацию
                log_config(self.config_path)

                # Шаг 1: Загрузка данных
                self.logger.info("Шаг 1: Загрузка данных")
                data_loader = DataLoader(
                    data_path=self.config.get('data_path'),
                    logger=self.logger
                )
                data = data_loader.load_data()

                if not data_loader.validate_data(data):
                    self.logger.error("Ошибка валидации данных")
                    return {'status': 'error', 'message': 'Ошибка валидации данных'}

                mlflow.log_param("data_size", len(data))

                # Шаг 2: Извлечение признаков
                self.logger.info("Шаг 2: Извлечение признаков")
                feature_extractor = FeatureExtractor(
                    cache_dir=self.config.get('cache_dir', 'data/cache'),
                    dimension_reduction=self.config.get('dimension_reduction', 'pca'),
                    n_components=self.config.get('n_components', 50),
                    batch_size=self.config.get('batch_size', 32),
                    device='cuda' if self.use_gpu else 'cpu',
                    logger=self.logger
                )
                features = feature_extractor.fit_transform(data)

                mlflow.log_param("n_features_raw", features.shape[1])

                # Шаг 3: Предобработка данных
                self.logger.info("Шаг 3: Предобработка данных")

                # 3.1 Обработка пропусков
                missing_handler = MissingValuesHandler(
                    strategy=self.config.get('missing_strategy', 'zeros'),
                    logger=self.logger
                )
                features_no_missing = missing_handler.fit_transform(features)

                # 3.2 Масштабирование
                scaler = FeatureScaler(
                    scaler_type=self.config.get('scaler_type', 'standard'),
                    scale_embeddings=self.config.get('scale_embeddings', False),
                    logger=self.logger
                )
                features_scaled = scaler.fit_transform(features_no_missing)

                # 3.3 Бинаризация целевой переменной
                target_binarizer = TargetBinarizer(
                    threshold=self.config.get('target_threshold', 100),
                    threshold_strategy=self.config.get('threshold_strategy', 'fixed'),
                    output_dir=self.output_dir / 'target_analysis',
                    logger=self.logger
                )
                target_column = self.config.get('target_column', 'tx_count')

                y_binary = target_binarizer.fit_transform(data[target_column])

                mlflow.log_param("target_threshold", target_binarizer.get_threshold())
                mlflow.log_param("positive_class_ratio", y_binary.mean())

                # Шаг 4: Корреляционный анализ признаков
                self.logger.info("Шаг 4: Корреляционный анализ признаков")
                correlation_selector = CorrelationSelector(
                    correlation_threshold=self.config.get('correlation_threshold', 0.95),
                    output_dir=self.output_dir / 'correlation_analysis',
                    logger=self.logger
                )
                X_corr = correlation_selector.fit_transform(features_scaled)

                mlflow.log_param("n_features_after_correlation", X_corr.shape[1])

                # Шаг 5: Отбор признаков с помощью Boruta
                self.logger.info("Шаг 5: Отбор признаков с помощью Boruta")
                boruta_selector = BorutaSelector(
                    n_estimators=self.config.get('boruta_n_estimators', 100),
                    max_iter=self.config.get('boruta_max_iter', 100),
                    output_dir=self.output_dir / 'boruta_analysis',
                    random_state=self.config.get('random_state', 42),
                    logger=self.logger
                )
                X_selected = boruta_selector.fit_transform(X_corr, y_binary)

                mlflow.log_param("n_features_final", X_selected.shape[1])

                # Шаг 6: Разделение на обучающую и тестовую выборки
                self.logger.info("Шаг 6: Разделение данных")
                data_splitter = DataSplitter(
                    test_size=self.config.get('test_size', 0.2),
                    random_state=self.config.get('random_state', 42),
                    stratify=True,
                    logger=self.logger
                )
                X_train, X_test, y_train, y_test = data_splitter.train_test_split(X_selected, y_binary)

                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))

                # Шаг 7: Обучение модели
                self.logger.info("Шаг 7: Обучение модели")
                model_trainer = ModelTrainer(
                    time_budget=self.config.get('time_budget', 3600),
                    estimator_list=self.config.get('estimator_list',
                                                   ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree', 'mlp']),
                    metric=self.config.get('metric', 'average_precision'),
                    n_folds=self.config.get('n_folds', 5),
                    ensemble_type=self.config.get('ensemble_type', 'stack'),
                    use_gpu=self.use_gpu,
                    output_dir=self.output_dir,
                    random_state=self.config.get('random_state', 42),
                    logger=self.logger
                )
                model_trainer.train(X_train, y_train)

                # Шаг 8: Оценка модели
                self.logger.info("Шаг 8: Оценка модели")
                metrics = model_trainer.evaluate(X_test, y_test)

                # Получаем предсказания для тестовой выборки
                y_pred_proba = model_trainer.automl.predict_proba(X_test.values)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)

                # Рассчитываем расширенные метрики
                metrics_calculator = ClassificationMetrics(logger=self.logger)
                all_metrics = metrics_calculator.get_all_metrics(y_test, y_pred, y_pred_proba)

                # Сохраняем отчет с метриками
                metrics_report = metrics_calculator.format_metrics_for_report(all_metrics['basic_metrics'])
                metrics_report_path = self.output_dir / 'metrics_report.txt'
                with open(metrics_report_path, 'w', encoding='utf-8') as f:
                    f.write(metrics_report)

                # Логируем метрики в MLflow
                log_model_metrics(all_metrics['basic_metrics'])

                # Шаг 9: Визуализация и интерпретация
                self.logger.info("Шаг 9: Визуализация и интерпретация результатов")
                visualizer = ModelVisualizer(
                    output_dir=self.output_dir / 'visualizations',
                    logger=self.logger
                )

                # Строим различные визуализации
                visualizer.plot_precision_recall_curve(y_test, y_pred_proba)
                visualizer.plot_roc_curve(y_test, y_pred_proba)
                f1_plot_path, best_threshold = visualizer.plot_f1_threshold(y_test, y_pred_proba)
                visualizer.plot_confusion_matrix(y_test, y_pred)

                # Визуализация с оптимальным порогом
                y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
                visualizer.plot_confusion_matrix(
                    y_test, y_pred_optimal,
                    filename='confusion_matrix_optimal_threshold.png'
                )

                # Важность признаков и SHAP
                feature_importance = model_trainer.get_feature_importance()
                if not feature_importance.empty:
                    visualizer.plot_feature_importance(feature_importance)

                visualizer.plot_lift_curve(y_test, y_pred_proba)

                # SHAP анализ, если доступен
                try:
                    visualizer.plot_shap_summary(model_trainer.automl.model.estimator, X_test)
                except Exception as e:
                    self.logger.warning(f"Не удалось построить SHAP диаграмму: {e}")

                # Шаг 10: Сохранение модели
                self.logger.info("Шаг 10: Сохранение модели")
                model_path = model_trainer.save_model(f"model_{self.timestamp}.joblib")

                # Завершение
                execution_time = time.time() - start_time
                self.logger.info(f"Пайплайн успешно выполнен за {execution_time:.2f} секунд")

                # Логируем артефакты
                mlflow.log_artifact(str(self.output_dir / 'pipeline.log'))
                mlflow.log_artifact(str(metrics_report_path))

                for artifact in (self.output_dir / 'visualizations').glob('*.png'):
                    mlflow.log_artifact(str(artifact), "visualizations")

                mlflow.log_metric("execution_time", execution_time)

                return {
                    'status': 'success',
                    'model_path': str(model_path),
                    'metrics': metrics,
                    'execution_time': execution_time,
                    'output_dir': str(self.output_dir)
                }

            except Exception as e:
                self.logger.exception(f"Ошибка при выполнении пайплайна: {e}")
                mlflow.log_param("error", str(e))

                return {
                    'status': 'error',
                    'message': str(e),
                    'output_dir': str(self.output_dir)
                }