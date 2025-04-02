import argparse
import sys
from datetime import datetime

from src.pipeline.pipeline import Pipeline
from src.utils.logger import setup_logger


def parse_arguments():
    """
    Парсинг аргументов командной строки.

    Returns:
        argparse.Namespace: Объект с аргументами
    """
    parser = argparse.ArgumentParser(description='Обучение модели классификации твитов')

    parser.add_argument('--config', type=str, required=True,
                        help='Путь к файлу конфигурации')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Директория для сохранения результатов (по умолчанию определяется из конфигурации)')

    parser.add_argument('--experiment-name', type=str, default='tweet_classification',
                        help='Название эксперимента в MLflow')

    parser.add_argument('--dimension-reduction', type=str, choices=['pca', 'tsne'], default='pca',
                        help='Метод снижения размерности для эмбеддингов (pca или tsne)')

    parser.add_argument('--missing-strategy', type=str, choices=['zeros', 'keep_nan'], default='zeros',
                        help='Стратегия обработки пропущенных значений (zeros или keep_nan)')

    parser.add_argument('--target-threshold', type=int, default=100,
                        help='Пороговое значение для бинаризации целевой переменной')

    parser.add_argument('--correlation-threshold', type=float, default=0.95,
                        help='Пороговое значение корреляции для отбора признаков')

    parser.add_argument('--time-budget', type=int, default=3600,
                        help='Бюджет времени на обучение модели (в секундах)')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Отключить использование GPU')

    parser.add_argument('--verbose', action='store_true',
                        help='Включить подробный вывод логов')

    return parser.parse_args()


def main():
    """
    Основная функция скрипта.
    """
    # Парсим аргументы командной строки
    args = parse_arguments()

    # Настраиваем логирование
    log_level = 'DEBUG' if args.verbose else 'INFO'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger(
        log_file=f'logs/pipeline_{timestamp}.log',
        level=log_level
    )

    logger.info("Запуск скрипта обучения модели классификации твитов")
    logger.info(f"Используемый файл конфигурации: {args.config}")

    try:
        # Обновляем конфигурацию из аргументов командной строки
        pipeline = Pipeline(
            config_path=args.config,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )

        # Обновляем параметры из командной строки
        if args.dimension_reduction:
            pipeline.config.set('dimension_reduction', args.dimension_reduction)

        if args.missing_strategy:
            pipeline.config.set('missing_strategy', args.missing_strategy)

        if args.target_threshold:
            pipeline.config.set('target_threshold', args.target_threshold)

        if args.correlation_threshold:
            pipeline.config.set('correlation_threshold', args.correlation_threshold)

        if args.time_budget:
            pipeline.config.set('time_budget', args.time_budget)

        if args.no_gpu:
            pipeline.config.set('use_gpu', False)

        # Запускаем пайплайн
        results = pipeline.run()

        # Выводим результаты
        if results['status'] == 'success':
            logger.info("Пайплайн успешно выполнен")
            logger.info(f"Модель сохранена в {results['model_path']}")
            logger.info(f"Метрики: {results['metrics']}")
            logger.info(f"Время выполнения: {results['execution_time']:.2f} секунд")
            logger.info(f"Результаты сохранены в {results['output_dir']}")
            return 0
        else:
            logger.error(f"Ошибка при выполнении пайплайна: {results['message']}")
            return 1

    except Exception as e:
        logger.exception(f"Необработанная ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())