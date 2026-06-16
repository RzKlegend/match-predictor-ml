# Match Predictor (ML)

Minimal project that trains a small RandomForest classifier to predict match outcome (home win / draw / away win) from historical international football results.

Usage:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. If you have the Kaggle dataset (`football_data.csv`) place it in the project root. The script will fall back to a sample dataset if missing.

3. Run the predictor:

```bash
python match_predictor.py
```
