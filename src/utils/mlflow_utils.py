import mlflow
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def setup_mlflow(
        experiment_name: str = 'tweet_classification',
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
) -> mlflow.tracking.MlflowClient:
    """
    Настройка MLflow для отслеживания экспериментов.

    Args:
        experiment_name: Название эксперимента
        tracking_uri: URI для MLflow сервера
        artifact_location: Путь для хранения артефактов

    Returns:
        mlflow.tracking.MlflowClient: Клиент MLflow
    """
    # Настраиваем URI для отслеживания
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Создаем или получаем эксперимент
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        if artifact_location:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # Создаем клиент
    client = mlflow.tracking.MlflowClient()

    logging.info(f"MLflow настроен для эксперимента '{experiment_name}' (ID: {experiment_id})")
    return client


def log_config(config_path: Union[str, Path]) -> None:
    """
    Логирование конфигурации в MLflow.

    Args:
        config_path: Путь к файлу конфигурации
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logging.warning(f"Файл конфигурации не найден: {config_path}")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Логируем параметры
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                mlflow.log_param(key, str(value))
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, int, float, bool)):
                        mlflow.log_param(f"{key}.{subkey}", subvalue)

        # Сохраняем файл конфигурации как артефакт
        mlflow.log_artifact(config_path)

    except Exception as e:
        logging.error(f"Ошибка при логировании конфигурации: {e}")


def log_model_metrics(metrics: Dict[str, Union[float, int, str]], step: Optional[int] = None) -> None:
    """
    Логирование метрик модели в MLflow.

    Args:
        metrics: Словарь с метриками
        step: Шаг (итерация)
    """
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (float, int)):
            mlflow.log_metric(metric_name, metric_value, step=step)


def log_model_params(params: Dict[str, Any]) -> None:
    """
    Логирование параметров модели в MLflow.

    Args:
        params: Словарь с параметрами
    """
    for param_name, param_value in params.items():
        if isinstance(param_value, (str, int, float, bool)):
            mlflow.log_param(param_name, param_value)
        else:
            # Для сложных типов конвертируем в строку
            mlflow.log_param(param_name, str(param_value))


def log_figure(figure_or_path: Union[plt.Figure, str, Path], artifact_path: Optional[str] = None) -> None:
    """
    Логирование рисунка (фигуры) в MLflow.

    Args:
        figure_or_path: Объект рисунка Matplotlib или путь к файлу
        artifact_path: Директория в artifacts для сохранения
    """
    try:
        if isinstance(figure_or_path, plt.Figure):
            # Если передан объект фигуры, сохраняем его во временный файл
            import tempfile
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                figure_or_path.savefig(tmp_path, dpi=300, bbox_inches='tight')

            # Логируем артефакт
            mlflow.log_artifact(tmp_path, artifact_path)

            # Удаляем временный файл
            tmp_path.unlink(missing_ok=True)
        else:
            # Если передан путь, логируем его как артефакт
            path = Path(figure_or_path)
            if path.exists():
                mlflow.log_artifact(path, artifact_path)
            else:
                logging.warning(f"Файл рисунка не найден: {path}")
    except Exception as e:
        logging.error(f"Ошибка при логировании рисунка: {e}")


def log_dataframe(
        df: pd.DataFrame,
        filename: str,
        format: str = 'csv',
        artifact_path: Optional[str] = None
) -> None:
    """
    Логирование DataFrame в MLflow.

    Args:
        df: DataFrame для логирования
        filename: Имя файла
        format: Формат выходного файла ('csv', 'json', 'html', 'pickle')
        artifact_path: Директория в artifacts для сохранения
    """
    try:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / filename

            # Сохраняем DataFrame в указанном формате
            if format.lower() == 'csv':
                df.to_csv(tmp_path, index=True)
            elif format.lower() == 'json':
                df.to_json(tmp_path, orient='records', lines=True)
            elif format.lower() == 'html':
                df.to_html(tmp_path, index=True)
            elif format.lower() == 'pickle':
                df.to_pickle(tmp_path)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")

            # Логируем артефакт
            mlflow.log_artifact(tmp_path, artifact_path)

    except Exception as e:
        logging.error(f"Ошибка при логировании DataFrame: {e}")


def log_feature_importance(
        feature_importance: pd.DataFrame,
        top_n: int = 20,
        artifact_path: Optional[str] = None
) -> None:
    """
    Логирование важности признаков в MLflow.

    Args:
        feature_importance: DataFrame с важностью признаков
        top_n: Количество топ-признаков для отображения
        artifact_path: Директория в artifacts для сохранения
    """
    if feature_importance.empty:
        logging.warning("DataFrame с важностью признаков пуст")
        return

    try:
        # Сортируем по убыванию важности
        sorted_importance = feature_importance.sort_values(
            'Importance' if 'Importance' in feature_importance.columns else feature_importance.columns[1],
            ascending=False
        )

        # Берем топ-N признаков
        top_features = sorted_importance.head(top_n)

        # Логируем DataFrame
        log_dataframe(top_features, 'feature_importance.csv', format='csv', artifact_path=artifact_path)

        # Создаем и логируем визуализацию
        plt.figure(figsize=(12, 8))
        feature_col = 'Feature' if 'Feature' in top_features.columns else top_features.columns[0]
        importance_col = 'Importance' if 'Importance' in top_features.columns else top_features.columns[1]

        plt.barh(top_features[feature_col], top_features[importance_col])
        plt.xlabel('Важность признака')
        plt.ylabel('Признак')
        plt.title('Топ важных признаков')
        plt.tight_layout()

        log_figure(plt.gcf(), artifact_path)
        plt.close()

    except Exception as e:
        logging.error(f"Ошибка при логировании важности признаков: {e}")


def log_confusion_matrix(
        confusion_matrix: List[List[int]],
        labels: Optional[List[str]] = None,
        artifact_path: Optional[str] = None
) -> None:
    """
    Логирование матрицы ошибок в MLflow.

    Args:
        confusion_matrix: Матрица ошибок
        labels: Метки классов
        artifact_path: Директория в artifacts для сохранения
    """
    try:
        import numpy as np
        import seaborn as sns

        cm = np.array(confusion_matrix)
        if labels is None:
            labels = [str(i) for i in range(len(cm))]

        # Создаем визуализацию
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Предсказанные метки')
        plt.ylabel('Истинные метки')
        plt.title('Матрица ошибок')

        # Логируем как артефакт
        log_figure(plt.gcf(), artifact_path)
        plt.close()

        # Также логируем как параметры
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                mlflow.log_metric(f"confusion_matrix_{true_label}_{pred_label}", cm[i, j])

    except Exception as e:
        logging.error(f"Ошибка при логировании матрицы ошибок: {e}")


def save_experiment_summary(output_dir: Union[str, Path], run_id: Optional[str] = None) -> None:
    """
    Сохранение сводки эксперимента в файл.

    Args:
        output_dir: Директория для сохранения сводки
        run_id: ID запуска MLflow (если None, используется текущий активный запуск)
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Получаем данные текущего запуска
        if run_id is None:
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(mlflow.active_run().info.run_id if mlflow.active_run() else None)
        else:
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(run_id)

        # Собираем информацию о запуске
        run_data = {
            'Run ID': run_info.info.run_id,
            'Experiment ID': run_info.info.experiment_id,
            'Status': run_info.info.status,
            'Start Time': run_info.info.start_time,
            'End Time': run_info.info.end_time,
            'Parameters': run_info.data.params,
            'Metrics': run_info.data.metrics,
            'Tags': run_info.data.tags
        }

        # Сохраняем в YAML-файл
        summary_path = output_dir / 'experiment_summary.yml'
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(run_data, f, default_flow_style=False, allow_unicode=True)

        logging.info(f"Сводка эксперимента сохранена в {summary_path}")

    except Exception as e:
        logging.error(f"Ошибка при сохранении сводки эксперимента: {e}")