import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, log_loss, classification_report
)


class ClassificationMetrics:
    """
    Класс для расчета и анализа метрик классификации.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация вычислителя метрик.

        Args:
            logger: Объект логгера
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {}
        self.thresholds = {}

    def calculate_basic_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Расчет основных метрик классификации.

        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов
            y_prob: Предсказанные вероятности (опционально)

        Returns:
            Dict[str, float]: Словарь с метриками
        """
        metrics = {}

        # Проверяем, что данные в правильном формате
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Рассчитываем метрики качества
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Рассчитываем метрики, требующие вероятностей
        if y_prob is not None:
            y_prob = np.asarray(y_prob)

            # Проверяем формат y_prob
            if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                # Если y_prob имеет формат [n_samples, 2], берем вероятности для класса 1
                y_prob = y_prob[:, 1]

            metrics['log_loss'] = log_loss(y_true, y_prob)
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)

        # Рассчитываем матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Дополнительные метрики из матрицы ошибок
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        self.metrics.update(metrics)
        return metrics

    def calculate_threshold_metrics(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            thresholds: Optional[List[float]] = None
    ) -> Dict[str, Dict[float, Dict[str, float]]]:
        """
        Расчет метрик для разных порогов вероятности.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            thresholds: Список порогов вероятности (если None, используются стандартные)

        Returns:
            Dict[str, Dict[float, Dict[str, float]]]: Словарь с метриками для разных порогов
        """
        # Проверяем, что данные в правильном формате
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Проверяем формат y_prob
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        # Если пороги не указаны, создаем стандартный набор
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19)

        threshold_metrics = {'thresholds': {}}
        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds:
            # Применяем порог
            y_pred = (y_prob >= threshold).astype(int)

            # Рассчитываем метрики
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

            # Сохраняем метрики для порога
            threshold_metrics['thresholds'][threshold] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }

            # Обновляем лучший порог по F1
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        threshold_metrics['best_threshold'] = best_threshold
        threshold_metrics['best_f1'] = best_f1

        self.thresholds = threshold_metrics
        return threshold_metrics

    def calculate_curves(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray
    ) -> Dict[str, Tuple]:
        """
        Расчет ROC-кривой и PR-кривой.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности

        Returns:
            Dict[str, Tuple]: Словарь с данными для кривых
        """
        # Проверяем, что данные в правильном формате
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Проверяем формат y_prob
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        # Рассчитываем ROC-кривую
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # Рассчитываем PR-кривую
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        curves = {
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds,
                'auc': roc_auc
            },
            'pr_curve': {
                'precision': precision,
                'recall': recall,
                'thresholds': pr_thresholds,
                'auc': pr_auc
            }
        }

        return curves

    def calculate_lift_curve(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Расчет Lift-кривой.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            n_bins: Количество бинов

        Returns:
            Dict[str, np.ndarray]: Словарь с данными для Lift-кривой
        """
        # Проверяем, что данные в правильном формате
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Проверяем формат y_prob
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        # Создаем DataFrame с предсказаниями и истинными метками
        df = pd.DataFrame({
            'y_true': y_true,
            'y_prob': y_prob
        })

        # Вычисляем общую долю положительных примеров
        overall_positive_rate = y_true.mean()

        # Сортируем по вероятностям (по убыванию)
        df = df.sort_values('y_prob', ascending=False)

        # Разбиваем на бины
        df['bin'] = pd.qcut(df.index, n_bins, labels=False)

        # Считаем lift для каждого бина
        lift_values = []
        bin_positive_rates = []
        cumulative_lifts = []
        cumulative_positive_rates = []

        for bin_idx in range(n_bins):
            bin_df = df[df['bin'] == bin_idx]
            bin_positive_rate = bin_df['y_true'].mean()
            lift = bin_positive_rate / overall_positive_rate

            lift_values.append(lift)
            bin_positive_rates.append(bin_positive_rate)

            # Для кумулятивного лифта
            cumulative_df = df[df['bin'] <= bin_idx]
            cumulative_positive_rate = cumulative_df['y_true'].mean()
            cumulative_lift = cumulative_positive_rate / overall_positive_rate

            cumulative_lifts.append(cumulative_lift)
            cumulative_positive_rates.append(cumulative_positive_rate)

        return {
            'lift_values': np.array(lift_values),
            'bin_positive_rates': np.array(bin_positive_rates),
            'cumulative_lifts': np.array(cumulative_lifts),
            'cumulative_positive_rates': np.array(cumulative_positive_rates),
            'bin_percentages': np.linspace(100 / n_bins, 100, n_bins)
        }

    def get_classification_report(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            target_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Получение подробного отчета о классификации.

        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов
            target_names: Названия классов

        Returns:
            Dict[str, Dict[str, float]]: Отчет о классификации
        """
        if target_names is None:
            target_names = ['class_0', 'class_1']

        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        return report

    def get_optimal_threshold(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            metric: str = 'f1',
            n_thresholds: int = 100
    ) -> Tuple[float, float]:
        """
        Определение оптимального порога по заданной метрике.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            metric: Метрика для оптимизации ('f1', 'precision', 'recall', 'accuracy')
            n_thresholds: Количество порогов для проверки

        Returns:
            Tuple[float, float]: Оптимальный порог и значение метрики
        """
        # Проверяем, что данные в правильном формате
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Проверяем формат y_prob
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        # Создаем список порогов
        thresholds = np.linspace(0.01, 0.99, n_thresholds)

        best_metric_value = 0
        best_threshold = 0.5

        # Словарь функций метрик
        metric_functions = {
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score,
            'accuracy': accuracy_score
        }

        if metric not in metric_functions:
            raise ValueError(f"Неподдерживаемая метрика: {metric}. Допустимые метрики: {list(metric_functions.keys())}")

        metric_function = metric_functions[metric]

        for threshold in thresholds:
            # Применяем порог
            y_pred = (y_prob >= threshold).astype(int)

            # Рассчитываем метрику
            if metric in ['precision', 'recall', 'f1']:
                metric_value = metric_function(y_true, y_pred, zero_division=0)
            else:
                metric_value = metric_function(y_true, y_pred)

            # Обновляем лучший порог
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold

        return best_threshold, best_metric_value

    def get_all_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: Optional[np.ndarray] = None,
            calculate_curves: bool = True,
            calculate_threshold_metrics: bool = True,
            n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Расчет всех метрик классификации.

        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов
            y_prob: Предсказанные вероятности (опционально)
            calculate_curves: Рассчитывать ли кривые (ROC, PR)
            calculate_threshold_metrics: Рассчитывать ли метрики для разных порогов
            n_bins: Количество бинов для Lift-кривой

        Returns:
            Dict[str, Any]: Словарь со всеми метриками
        """
        # Рассчитываем основные метрики
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)

        all_metrics = {'basic_metrics': basic_metrics}

        # Если предоставлены вероятности, рассчитываем дополнительные метрики
        if y_prob is not None:
            # Рассчитываем метрики для разных порогов
            if calculate_threshold_metrics:
                threshold_metrics = self.calculate_threshold_metrics(y_true, y_prob)
                all_metrics['threshold_metrics'] = threshold_metrics

            # Рассчитываем кривые
            if calculate_curves:
                curves = self.calculate_curves(y_true, y_prob)
                all_metrics['curves'] = curves

            # Рассчитываем Lift-кривую
            lift_curve = self.calculate_lift_curve(y_true, y_prob, n_bins)
            all_metrics['lift_curve'] = lift_curve

        # Получаем подробный отчет
        report = self.get_classification_report(y_true, y_pred)
        all_metrics['classification_report'] = report

        return all_metrics

    def format_metrics_for_report(self, metrics: Optional[Dict] = None) -> str:
        """
        Форматирование метрик для отчета.

        Args:
            metrics: Словарь с метриками (если None, используются последние рассчитанные)

        Returns:
            str: Отформатированный отчет
        """
        if metrics is None:
            metrics = self.metrics

        if not metrics:
            return "Метрики не рассчитаны"

        lines = ["=== ОТЧЕТ ПО МЕТРИКАМ МОДЕЛИ ===\n"]

        # Основные метрики
        lines.append("Основные метрики:")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            if metric_name in metrics:
                lines.append(f"  {metric_name}: {metrics[metric_name]:.4f}")

        # Матрица ошибок
        if all(key in metrics for key in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']):
            lines.append("\nМатрица ошибок:")
            lines.append(f"  [[{metrics['true_negatives']}, {metrics['false_positives']}],")
            lines.append(f"   [{metrics['false_negatives']}, {metrics['true_positives']}]]")

            # Дополнительные метрики из матрицы ошибок
            lines.append("\nДополнительные метрики:")
            for metric_name in ['specificity', 'npv']:
                if metric_name in metrics:
                    lines.append(f"  {metric_name}: {metrics[metric_name]:.4f}")

        # Лучший порог
        if 'thresholds' in self.thresholds:
            lines.append("\nЛучший порог (по F1):")
            lines.append(f"  Порог: {self.thresholds['best_threshold']:.4f}")
            lines.append(f"  F1: {self.thresholds['best_f1']:.4f}")

        # Описание метрик
        lines.append("\n=== Описание метрик ===")
        lines.append("accuracy: доля правильно классифицированных объектов")
        lines.append(
            "precision: доля объектов, которые действительно положительные среди всех объектов, отнесенных моделью к положительным")
        lines.append("recall: доля положительных объектов, которые были правильно идентифицированы моделью")
        lines.append("f1: гармоническое среднее между precision и recall")
        lines.append("roc_auc: площадь под ROC-кривой")
        lines.append("pr_auc: площадь под PR-кривой")
        lines.append("specificity: доля правильно идентифицированных отрицательных объектов")
        lines.append(
            "npv: доля правильно идентифицированных отрицательных объектов среди всех отрицательных предсказаний")

        return "\n".join(lines)