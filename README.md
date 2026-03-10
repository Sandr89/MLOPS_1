# Лабораторна робота №1–2 - MLOps (Варіант 19)

**Twitter Sentiment Analysis (Hate Speech)** — виявлення расистських/сексистських твітів.

## Структура проєкту

```
laba1_v2/
├── .gitignore
├── requirements.txt
├── README.md
├── dvc.yaml              # DVC пайплайн (prepare → train)
├── dvc.lock              # Фіксація версій
├── data/
│   ├── raw/              # train.csv (DVC: data/raw/train.csv.dvc)
│   ├── prepared/         # вихід prepare (train.csv, test.csv)
│   └── models/           # вихід train (model.joblib)
├── src/
│   ├── prepare.py        # Етап підготовки даних
│   ├── train.py          # Етап навчання + MLflow
│   └── preprocess.py     # Очистка тексту
├── notebooks/
│   └── 01_eda.ipynb      # EDA
└── mlruns/               # Логи MLflow
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

### DVC-пайплайн (Lab 2)
Дані та модель версіонуються через DVC. Повний цикл:
```bash
# Після клону: завантажити дані з remote
dvc pull

# Запустити пайплайн (prepare → train)
dvc repro
```
Повторний `dvc repro` без змін пропускає кроки (кеш). Зміна коду або даних перезапускає лише потрібні стадії.

Одноразове налаштування (вже зроблено в репо):
- `dvc init` — ініціалізація DVC
- `dvc remote add -d mylocal ../dvc_storage` — локальне сховище
- `dvc add data/raw/train.csv` — версіонування сирих даних
- `dvc push` — відправити дані в remote

### EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Навчання (вручну, після prepare)
```bash
python src/train.py data/prepared data/models --max_depth 10
```
Або 5+ експериментів для аналізу overfitting: змінювати `--max_depth` (2, 5, 10, 20, 50).

### MLflow UI
**Важливо:** Запускати з директорії проєкту, або використати скрипт:
```bash
# З директорії проєкту:
mlflow ui

# Або запустити run_mlflow_ui.bat (Windows)
```
Відкрити http://127.0.0.1:5000 → Compare запусків.

## CLI-аргументи (train.py)

| Аргумент | Опис | За замовчуванням |
|----------|------|------------------|
| prepared_dir | Шлях до data/prepared | — |
| model_dir | Шлях для збереження моделі | — |
| --max_features | Макс. ознак TF-IDF | 5000 |
| --n_estimators | Кількість дерев | 100 |
| --max_depth | Глибина дерева | 10 |
| --random_state | Seed | 42 |
