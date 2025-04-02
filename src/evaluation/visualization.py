import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    f1_score, confusion_matrix, classification_report
)
import shap


class ModelVisualizer:
    """
    Класс для визуализации результатов работы модели.
    """

    def __init__(
            self,
            output_dir: Union[str, Path] = 'visualizations',
            fig_size: Tuple[int, int] = (10, 8),
            dpi: int = 300,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация визуализатора.

        Args:
            output_dir: Директория для сохранения визуализаций
            fig_size: Размер фигуры по умолчанию
            dpi: Разрешение изображений
            logger: Объект логгера
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_size = fig_size
        self.dpi = dpi
        self.logger = logger or logging.getLogger(__name__)

    def plot_precision_recall_curve(
            self,
            y_true: Union[np.ndarray, pd.Series],
            y_prob: np.ndarray,
            filename: str = 'precision_recall_curve.png'
    ) -> Path:
        """
        Построение кривой точность-полнота.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненному изображению
        """
        self.logger.info("Построение кривой точность-полнота")

        # Преобразуем Series в numpy array при необходимости
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        # Рассчитываем кривую PR
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=self.fig_size)
        plt.plot(recall, precision, lw=2, label=f'PR кривая (AUC = {pr_auc:.3f})')
        plt.xlabel('Полнота (Recall)')
        plt.ylabel('Точность (Precision)')
        plt.title('Кривая точность-полнота')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Добавляем базовую линию случайного классификатора
        plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], 'r--',
                 label=f'Случайный классификатор (AUC = {np.mean(y_true):.3f})')
        plt.legend(loc='best')

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Кривая точность-полнота сохранена в {output_path}")
        return output_path

    def plot_roc_curve(
            self,
            y_true: Union[np.ndarray, pd.Series],
            y_prob: np.ndarray,
            filename: str = 'roc_curve.png'
    ) -> Path:
        """
        Построение ROC-кривой.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненному изображению
        """
        self.logger.info("Построение ROC-кривой")

        # Преобразуем Series в numpy array при необходимости
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        # Рассчитываем ROC-кривую
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=self.fig_size)
        plt.plot(fpr, tpr, lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Случайный классификатор')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"ROC-кривая сохранена в {output_path}")
        return output_path

    def plot_f1_threshold(
            self,
            y_true: Union[np.ndarray, pd.Series],
            y_prob: np.ndarray,
            n_thresholds: int = 100,
            filename: str = 'f1_threshold.png'
    ) -> Tuple[Path, float]:
        """
        Построение графика зависимости F1-меры от порога.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            n_thresholds: Количество порогов для проверки
            filename: Имя файла для сохранения

        Returns:
            Tuple[Path, float]: Путь к сохраненному изображению и оптимальный порог
        """
        self.logger.info("Построение графика зависимости F1-меры от порога")

        # Преобразуем Series в numpy array при необходимости
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        # Создаем список порогов
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        f1_scores = []

        # Рассчитываем F1-меру для каждого порога
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)

        # Находим оптимальный порог
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        # Строим график
        plt.figure(figsize=self.fig_size)
        plt.plot(thresholds, f1_scores, 'b-', lw=2)
        plt.axvline(x=best_threshold, color='r', linestyle='--',
                    label=f'Оптимальный порог = {best_threshold:.3f}, F1 = {best_f1:.3f}')
        plt.xlabel('Порог вероятности')
        plt.ylabel('F1-мера')
        plt.title('Зависимость F1-меры от порога вероятности')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"График F1-порог сохранен в {output_path}")
        self.logger.info(f"Оптимальный порог: {best_threshold:.3f}, F1: {best_f1:.3f}")
        return output_path, best_threshold

    def plot_confusion_matrix(
            self,
            y_true: Union[np.ndarray, pd.Series],
            y_pred: np.ndarray,
            normalize: bool = False,
            filename: str = 'confusion_matrix.png'
    ) -> Path:
        """
        Построение матрицы ошибок.

        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов
            normalize: Нормализовать ли значения матрицы
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненному изображению
        """
        self.logger.info("Построение матрицы ошибок")

        # Преобразуем Series в numpy array при необходимости
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        # Рассчитываем матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)

        # Нормализуем, если требуется
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Нормализованная матрица ошибок'
        else:
            fmt = 'd'
            title = 'Матрица ошибок'

        # Строим тепловую карту
        plt.figure(figsize=self.fig_size)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Класс 0', 'Класс 1'],
                    yticklabels=['Класс 0', 'Класс 1'])
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.title(title)

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Матрица ошибок сохранена в {output_path}")
        return output_path

    def plot_feature_importance(
            self,
            feature_importance: pd.DataFrame,
            top_n: int = 20,
            filename: str = 'feature_importance.png'
    ) -> Path:
        """
        Построение графика важности признаков.

        Args:
            feature_importance: DataFrame с важностью признаков
            top_n: Количество наиболее важных признаков для отображения
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненному изображению
        """
        self.logger.info(f"Построение графика важности признаков (top {top_n})")

        # Проверяем наличие данных
        if feature_importance.empty:
            self.logger.warning("Данные о важности признаков отсутствуют")
            return None

        # Определяем колонки с признаками и их важностью
        feature_col = 'Feature' if 'Feature' in feature_importance.columns else feature_importance.columns[0]
        importance_col = 'Importance' if 'Importance' in feature_importance.columns else feature_importance.columns[1]

        # Сортируем и берем top_n
        sorted_importance = feature_importance.sort_values(importance_col, ascending=False)
        if len(sorted_importance) > top_n:
            sorted_importance = sorted_importance.head(top_n)

        # Строим горизонтальную гистограмму
        plt.figure(figsize=(12, max(8, len(sorted_importance) * 0.3)))
        ax = sns.barplot(x=importance_col, y=feature_col, data=sorted_importance)
        plt.title(f'Топ-{top_n} важных признаков')
        plt.xlabel('Важность')
        plt.tight_layout()

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"График важности признаков сохранен в {output_path}")
        return output_path

    def plot_lift_curve(
            self,
            y_true: Union[np.ndarray, pd.Series],
            y_prob: np.ndarray,
            n_bins: int = 10,
            filename: str = 'lift_curve.png'
    ) -> Path:
        """
        Построение Lift-кривой.

        Args:
            y_true: Истинные метки классов
            y_prob: Предсказанные вероятности
            n_bins: Количество бинов для кривой
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненному изображению
        """
        self.logger.info("Построение Lift-кривой")

        # Преобразуем Series в numpy array при необходимости
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

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

        # Создаем фигуру с 2 подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Строим график для бинов
        bin_percentages = np.linspace(100 / n_bins, 100, n_bins)
        ax1.bar(bin_percentages, lift_values, width=100 / n_bins * 0.8)
        ax1.set_xlabel('Процентиль (по вероятности предсказания)')
        ax1.set_ylabel('Lift')
        ax1.set_title('Lift по бинам')
        ax1.axhline(y=1, color='r', linestyle='--', label='Базовый уровень')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Строим кумулятивный график
        ax2.plot(bin_percentages, cumulative_lifts, 'b-', marker='o')
        ax2.set_xlabel('Процентиль (по вероятности предсказания)')
        ax2.set_ylabel('Кумулятивный Lift')
        ax2.set_title('Кумулятивный Lift')
        ax2.axhline(y=1, color='r', linestyle='--', label='Базовый уровень')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Lift-кривая сохранена в {output_path}")
        return output_path

    def plot_shap_summary(
            self,
            model,
            X_test: pd.DataFrame,
            max_display: int = 20,
            filename: str = 'shap_summary.png'
    ) -> Optional[Path]:
        """
        Построение SHAP-диаграммы для модели.

        Args:
            model: Обученная модель
            X_test: Тестовые данные для объяснения
            max_display: Максимальное количество признаков для отображения
            filename: Имя файла для сохранения

        Returns:
            Optional[Path]: Путь к сохраненному изображению или None в случае ошибки
        """
        self.logger.info("Построение SHAP-диаграммы")

        try:
            # Создаем SHAP explainer в зависимости от типа модели
            model_type = type(model).__name__.lower()

            if 'lightgbm' in model_type or model_type == 'lgbm':
                explainer = shap.TreeExplainer(model)
            elif 'xgboost' in model_type:
                explainer = shap.TreeExplainer(model)
            elif 'catboost' in model_type:
                explainer = shap.TreeExplainer(model)
            elif 'randomforest' in model_type or 'rf' in model_type:
                explainer = shap.TreeExplainer(model)
            elif 'extratrees' in model_type or 'extra_tree' in model_type:
                explainer = shap.TreeExplainer(model)
            else:
                # Для других моделей используем KernelExplainer
                # Берем подвыборку для ускорения расчетов
                sample_size = min(100, len(X_test))
                background = shap.kmeans(X_test.values, 10).data
                explainer = shap.KernelExplainer(model.predict_proba, background)
                X_test = X_test.head(sample_size)

            # Вычисляем SHAP значения
            shap_values = explainer.shap_values(X_test)

            # Для моделей, которые возвращают значения для обоих классов, берем значения для класса 1
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]

            # Строим summary plot
            plt.figure(figsize=self.fig_size)
            shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)

            # Сохраняем изображение
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            self.logger.info(f"SHAP-диаграмма сохранена в {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Ошибка при построении SHAP-диаграммы: {e}")
            return None

    def plot_learning_curve(
            self,
            train_scores: List[float],
            val_scores: List[float],
            train_sizes: List[int],
            metric_name: str = 'Score',
            filename: str = 'learning_curve.png'
    ) -> Path:
        """
        Построение кривой обучения.

        Args:
            train_scores: Значения метрики на обучающей выборке
            val_scores: Значения метрики на валидационной выборке
            train_sizes: Размеры обучающей выборки
            metric_name: Название метрики
            filename: Имя файла для сохранения

        Returns:
            Path: Путь к сохраненному изображению
        """
        self.logger.info("Построение кривой обучения")

        plt.figure(figsize=self.fig_size)
        plt.plot(train_sizes, train_scores, 'o-', color='r', label=f'Обучающая выборка')
        plt.plot(train_sizes, val_scores, 'o-', color='g', label=f'Валидационная выборка')
        plt.xlabel('Размер обучающей выборки')
        plt.ylabel(metric_name)
        plt.title(f'Кривая обучения ({metric_name})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')

        # Сохраняем изображение
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Кривая обучения сохранена в {output_path}")
        return output_path