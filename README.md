# Лабораторні роботи 1-4 з MLOps

Проєкт присвячений класифікації hate speech у Twitter/X і поступово розширювався від EDA та baseline-моделі до DVC, MLflow, Hydra, Optuna і CI/CD для ML.

## Структура проєкту

```text
laba1_v2/
|-- .github/workflows/cml.yaml
|-- config/
|   |-- config.yaml
|   |-- model/
|   `-- hpo/
|-- data/
|   |-- raw/
|   |-- prepared/
|   `-- models/
|-- notebooks/
|   `-- 01_eda.ipynb
|-- src/
|   |-- preprocess.py
|   |-- prepare.py
|   |-- train.py
|   `-- optimize.py
|-- tests/
|   |-- test_pretrain.py
|   `-- test_posttrain.py
|-- dvc.yaml
|-- dvc.lock
`-- requirements.txt
```

## Встановлення

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Дані

- Основний датасет: `data/raw/train.csv`
- DVC використовується для відтворюваності pipeline
- Підготовлені спліти зберігаються в `data/prepared/train.csv` і `data/prepared/test.csv`

За потреби можна підтягнути артефакти DVC:

```bash
dvc pull
```

## DVC pipeline

Підготовка даних:

```bash
dvc repro prepare
```

Тренування baseline-моделі:

```bash
dvc repro train
```

Повний baseline-цикл:

```bash
dvc repro prepare train
```

HPO-етап з Hydra + Optuna:

```bash
dvc repro optimize
```

## Локальний запуск скриптів

Підготовка даних:

```bash
python src/prepare.py data/raw/train.csv data/prepared
```

Тренування baseline-моделі:

```bash
python src/train.py data/prepared data/models
```

Пошук гіперпараметрів:

```bash
python src/optimize.py
```

## Артефакти тренування

Після `train` етапу створюються:

- `data/models/model.joblib`
- `data/models/metrics.json`
- `data/models/confusion_matrix.png`
- `data/models/feature_importance.png`

Файл `metrics.json` містить як test-метрики для CI (`accuracy`, `f1`), так і деталізовані train/test значення.

## Тести

Швидкі pre-train перевірки:

```bash
python -m pytest tests/test_pretrain.py -q
```

Post-train перевірки артефактів і Quality Gate:

```bash
python -m pytest tests/test_posttrain.py -q
```

Повний прогін тестів:

```bash
python -m pytest -q
```

## Quality Gate

- Порогове значення задається через змінну середовища `F1_THRESHOLD`
- Значення за замовчуванням: `0.12`
- Перевірка виконується в `tests/test_posttrain.py`

Приклад локального запуску з явним порогом:

```bash
set F1_THRESHOLD=0.12
python -m pytest tests/test_posttrain.py -q
```

## CI/CD для ЛР4

Workflow знаходиться у `.github/workflows/cml.yaml` і виконує:

1. встановлення Python-залежностей
2. перевірку коду через `flake8`
3. перевірку форматування через `black --check`
4. запуск pre-train тестів
5. відтворення `prepare` і `train` через `dvc repro`
6. запуск post-train тестів
7. генерацію CML-звіту в Pull Request
8. завантаження артефактів моделі для `push` у `main`

CML-звіт для PR містить:

- таблицю числових метрик із `data/models/metrics.json`
- зображення `confusion_matrix.png`
- результат Quality Gate

## MLflow

Для локального перегляду експериментів:

```bash
python run_mlflow_ui.py
```

Після запуску UI відкривається за адресою [http://127.0.0.1:5000](http://127.0.0.1:5000).