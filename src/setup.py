from setuptools import setup, find_packages

setup(
    name="tweet-classification-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=1.10.0",
        "transformers>=4.12.0",
        "flaml>=1.0.0",
        "lightgbm>=4.6.0",
        "xgboost>=1.5.0",
        "catboost>=1.0.0",
        "shap>=0.40.0",
        "mlflow>=1.20.0",
        "boruta>=0.3.0",
        "emoji>=1.6.0",
        "pyyaml>=6.0.0",
        "requests>=2.26.0",
        "pillow>=8.3.0",
    ],
    python_requires=">=3.8",
)