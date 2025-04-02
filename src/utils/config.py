import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Config:
    """
    Класс для работы с конфигурационными файлами проекта.
    """

    def __init__(self, config_path: Union[str, Path], logger: Optional[logging.Logger] = None):
        """
        Инициализация объекта конфигурации.

        Args:
            config_path: Путь к файлу конфигурации (YAML)
            logger: Объект логгера
        """
        self.config_path = Path(config_path)
        self.logger = logger or logging.getLogger(__name__)
        self.config = {}

        self._load_config()

    def _load_config(self) -> None:
        """
        Загрузка конфигурации из файла.
        """
        if not self.config_path.exists():
            self.logger.error(f"Файл конфигурации не найден: {self.config_path}")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Загружена конфигурация из {self.config_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения из конфигурации по ключу.

        Args:
            key: Ключ для поиска в конфигурации
            default: Значение по умолчанию, если ключ не найден

        Returns:
            Any: Значение из конфигурации или значение по умолчанию
        """
        # Поддержка вложенных ключей через точку (e.g., "logging.level")
        if '.' in key:
            parts = key.split('.')
            curr = self.config
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return default
            return curr

        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Установка значения в конфигурацию.

        Args:
            key: Ключ для установки в конфигурации
            value: Значение для установки
        """
        # Поддержка вложенных ключей через точку
        if '.' in key:
            parts = key.split('.')
            curr = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            curr[parts[-1]] = value
        else:
            self.config[key] = value

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Сохранение конфигурации в файл.

        Args:
            output_path: Путь для сохранения конфигурации (если None, используется исходный путь)
        """
        save_path = Path(output_path) if output_path else self.config_path

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Конфигурация сохранена в {save_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении конфигурации: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Получение конфигурации в виде словаря.

        Returns:
            Dict[str, Any]: Словарь с конфигурацией
        """
        return self.config.copy()