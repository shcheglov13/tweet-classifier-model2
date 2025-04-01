import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union


class DataLoader:
    """
    Класс для загрузки и базовой валидации данных.
    """

    def __init__(self, data_path: Union[str, Path], logger: Optional[logging.Logger] = None):
        """
        Инициализация загрузчика данных.

        Args:
            data_path: Путь к файлу данных
            logger: Объект логгера
        """
        self.data_path = Path(data_path)
        self.logger = logger or logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """
        Загрузка данных из файла.

        Returns:
            pd.DataFrame: Загруженный датафрейм
        """
        self.logger.info(f"Загрузка данных из {self.data_path}")

        if self.data_path.suffix == '.csv':
            data = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.json':
            data = pd.read_json(self.data_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {self.data_path.suffix}")

        self.logger.info(f"Загружено {len(data)} записей")
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Базовая валидация структуры данных.

        Args:
            data: Датафрейм для валидации

        Returns:
            bool: Результат валидации
        """
        required_columns = ['id', 'text', 'tweet_type', 'tx_count']

        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Отсутствует обязательная колонка: {col}")
                return False

        # Проверка типов данных
        if not pd.api.types.is_numeric_dtype(data['tx_count']):
            self.logger.error("Колонка tx_count должна быть числового типа")
            return False

        return True