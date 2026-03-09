# Лабораторна робота №1 - MLOps (Варіант 19)

**Twitter Sentiment Analysis (Hate Speech)** — виявлення расистських/сексистських твітів.

## Структура проєкту

```
laba1_v2/
├── .gitignore
├── requirements.txt
├── README.md
├── download_data.py      # Скрипт завантаження датасету
├── data/
│   └── raw/              # train.csv (завантажити з Kaggle)
├── sample_data/          # train_sample.csv для тесту без завантаження
├── notebooks/
│   └── 01_eda.ipynb      # EDA
├── src/
│   ├── train.py          # Навчання + MLflow
│   └── preprocess.py     # Очистка тексту
├── mlruns/               # Логи MLflow
└── models/
```

## Встановлення

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Завантаження даних

1. Зареєструйтесь на [Kaggle](https://www.kaggle.com)
2. Створіть API token: Account → Create New API Token
3. Помістіть `kaggle.json` у `%USERPROFILE%\.kaggle\`
4. Запустіть: `python download_data.py`

Або завантажте вручну: [Twitter Sentiment Analysis Hatred Speech](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) → Download → розпакуйте `train.csv` у `data/raw/`

## Використання

### EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Навчання (5+ експериментів для аналізу overfitting)
```bash
python src/train.py --max_depth 2
python src/train.py --max_depth 5
python src/train.py --max_depth 10
python src/train.py --max_depth 20
python src/train.py --max_depth 50
```

### MLflow UI
**Важливо:** Запускати з директорії проєкту, або використати скрипт:
```bash
# З директорії проєкту:
mlflow ui

# Або запустити run_mlflow_ui.bat (Windows)
```
Відкрити http://127.0.0.1:5000 → Compare запусків.

## CLI-аргументи

| Аргумент | Опис | За замовчуванням |
|----------|------|------------------|
| --max_features | Макс. ознак TF-IDF | 5000 |
| --n_estimators | Кількість дерев | 100 |
| --max_depth | Глибина дерева | 10 |
| --random_state | Seed | 42 |
| --test_size | Частка тесту | 0.2 |
