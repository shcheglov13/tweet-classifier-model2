import pandas as pd
import logging
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import Patch


class BorutaSelector:
    """
    Класс для отбора признаков с помощью алгоритма Boruta.
    """

    def __init__(
            self,
            n_estimators: int = 100,
            perc: int = 90,
            max_iter: int = 100,
            embedding_prefixes: List[str] = ['text_emb_', 'quoted_text_emb_', 'image_emb_'],
            random_state: int = 42,
            output_dir: Optional[Path] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация селектора Boruta.

        Args:
            n_estimators: Количество деревьев в случайном лесу
            perc: Процентиль для определения порога значимости
            max_iter: Максимальное число итераций
            embedding_prefixes: Префиксы для идентификации колонок с эмбеддингами
            random_state: Состояние генератора случайных чисел
            output_dir: Директория для сохранения визуализаций
            logger: Объект логгера
        """
        self.n_estimators = n_estimators
        self.perc = perc
        self.max_iter = max_iter
        self.embedding_prefixes = embedding_prefixes
        self.random_state = random_state
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)

        self.boruta = None
        self.selected_features = []
        self.rejected_features = []
        self.tentative_features = []
        self.feature_ranks = None

    def _identify_embedding_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Идентификация колонок с эмбеддингами.

        Args:
            data: Датафрейм с признаками

        Returns:
            List[str]: Список колонок с эмбеддингами
        """
        embedding_columns = []
        for prefix in self.embedding_prefixes:
            embedding_columns.extend([col for col in data.columns if col.startswith(prefix)])

        return embedding_columns

    def _plot_feature_ranks(self, output_path: Optional[Path] = None):
        """
        Построение графика важности признаков.

        Args:
            output_path: Путь для сохранения изображения
        """
        if self.feature_ranks is None or len(self.feature_ranks) == 0:
            return

        # Сортируем признаки по убыванию важности
        plot_data = self.feature_ranks.sort_values('Rank')

        # Ограничиваем количество признаков для наглядности
        max_features = 30
        if len(plot_data) > max_features:
            plot_data = plot_data.head(max_features)

        plt.figure(figsize=(12, max(8, len(plot_data) * 0.3)))
        colors = ['green' if s == 'Accepted' else ('blue' if s == 'Tentative' else 'red') for s in
                  plot_data['Decision']]

        ax = sns.barplot(x='Rank', y='Feature', data=plot_data, palette=colors)
        plt.title('Важность признаков (Boruta)')
        plt.xlabel('Ранг')
        plt.ylabel('Признак')

        # Добавляем легенду
        legend_elements = [
            Patch(facecolor='green', label='Принят'),
            Patch(facecolor='blue', label='Под вопросом'),
            Patch(facecolor='red', label='Отклонен')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График важности признаков сохранен в {output_path}")

        plt.close()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BorutaSelector':
        """
        Обучение селектора Boruta.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной

        Returns:
            self: Возвращает экземпляр класса
        """
        self.logger.info("Начало отбора признаков с помощью Boruta")

        # Идентифицируем эмбеддинги
        embedding_columns = self._identify_embedding_columns(X)
        non_embedding_columns = [col for col in X.columns if col not in embedding_columns]

        # Разделяем данные на эмбеддинги и другие признаки
        X_non_embedding = X[non_embedding_columns].copy()

        # Создаем и обучаем Boruta только на неэмбеддинговых признаках
        rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
        self.boruta = BorutaPy(
            rf,
            n_estimators='auto',
            perc=self.perc,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0
        )

        self.logger.info(f"Обучение Boruta на {X_non_embedding.shape[1]} признаках")
        try:
            self.boruta.fit(X_non_embedding.values, y.values)

            # Сохраняем результаты
            self.feature_ranks = pd.DataFrame({
                'Feature': non_embedding_columns,
                'Rank': self.boruta.ranking_,
                'Support': self.boruta.support_,
                'Decision': ['Accepted' if s else 'Rejected' for s in self.boruta.support_]
            })

            # Отмечаем неопределенные признаки
            tentative_mask = self.boruta.support_weak_
            self.feature_ranks.loc[tentative_mask, 'Decision'] = 'Tentative'

            self.selected_features = list(self.feature_ranks[(self.feature_ranks['Support']) | self.boruta.support_weak_]['Feature'])
            self.rejected_features = list(self.feature_ranks[(~self.feature_ranks['Support']) & (~self.boruta.support_weak_)]['Feature'])
            self.tentative_features = list(self.feature_ranks[self.boruta.support_weak_]['Feature'])

            # Добавляем все эмбеддинги к отобранным признакам
            self.selected_features.extend(embedding_columns)

            self.logger.info(f"Отобрано {len(self.selected_features)} признаков (включая Tentative)")
            self.logger.info(f"Отклонено {len(self.rejected_features)} признаков")
            self.logger.info(f"Нерешенные признаки: {len(self.tentative_features)}")

            # Визуализация результатов
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._plot_feature_ranks(
                    self.output_dir / 'boruta_feature_ranks.png'
                )

        except Exception as e:
            self.logger.error(f"Ошибка при обучении Boruta: {e}")
            # В случае ошибки сохраняем все признаки
            self.selected_features = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование данных, оставляя только отобранные признаки.

        Args:
            X: Датафрейм с признаками

        Returns:
            pd.DataFrame: Датафрейм с отобранными признаками
        """
        if not self.selected_features:
            self.logger.warning("Нет отобранных признаков, возвращаю исходные данные")
            return X

        # Оставляем только те признаки, которые есть в данных
        selected = [col for col in self.selected_features if col in X.columns]

        self.logger.info(f"Оставляю {len(selected)} отобранных признаков из {X.shape[1]}")
        return X[selected]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Обучение селектора и преобразование данных.

        Args:
            X: Датафрейм с признаками
            y: Серия с целевой переменной

        Returns:
            pd. DataFrame: Датафрейм с отобранными признаками
        """
        return self.fit(X, y).transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Получение важности признаков.

        Returns:
            pd.DataFrame: Датафрейм с важностью признаков
        """
        if self.feature_ranks is None:
            self.logger.warning("Boruta не обучен")
            return pd.DataFrame()

        return self.feature_ranks.sort_values('Rank')