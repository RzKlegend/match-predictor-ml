"""Minimal ML-based match predictor.

Loads `football_data.csv` if present, trains a simple RandomForest to predict match outcome
(home win / draw / away win) using lightweight features derived from team historical stats.
If dataset is missing, falls back to a small sample.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_dataset(path='football_data.csv', max_rows=None):
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset from {path} with {len(df)} rows")
        if max_rows:
            df = df.tail(max_rows)
        return df
    except Exception as e:
        print(f"Could not load {path}: {e}\nUsing small sample dataset.")
        teams = ['Saudi Arabia', 'Uruguay', 'Brazil', 'Argentina', 'Germany', 'France']
        rows = []
        rng = np.random.RandomState(0)
        for i in range(500):
            h = rng.choice(teams)
            a = rng.choice([t for t in teams if t != h])
            hs = rng.poisson(1.5)
            as_ = rng.poisson(1.2)
            rows.append({'date': '2020-01-01', 'home_team': h, 'away_team': a, 'home_score': hs, 'away_score': as_, 'neutral': False})
        return pd.DataFrame(rows)


def prepare_features(df):
    # Drop matches without scores
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    # Target: 0=home win,1=draw,2=away win
    df['result'] = df.apply(lambda r: 0 if r['home_score']>r['away_score'] else (1 if r['home_score']==r['away_score'] else 2), axis=1)

    # Team-level aggregates
    teams = pd.concat([df[['home_team','home_score']].rename(columns={'home_team':'team','home_score':'gf'}),
                       df[['away_team','away_score']].rename(columns={'away_team':'team','away_score':'gf'})])
    team_gf = teams.groupby('team')['gf'].mean().to_dict()

    # Build features for each match
    X = []
    y = []
    for _, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        h_gf = team_gf.get(h, 1.0)
        a_gf = team_gf.get(a, 1.0)
        neutral = 1 if row.get('neutral', False) else 0
        X.append([h_gf, a_gf, h_gf - a_gf, neutral])
        y.append(row['result'])

    X = np.array(X)
    y = np.array(y)
    return X, y, team_gf


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("\nModel evaluation:\n", classification_report(y_test, preds, zero_division=0))
    return clf


def predict_match(clf, team_gf, home, away, neutral=False):
    h_gf = team_gf.get(home, 1.0)
    a_gf = team_gf.get(away, 1.0)
    feat = np.array([[h_gf, a_gf, h_gf - a_gf, 1 if neutral else 0]])
    prob = clf.predict_proba(feat)[0]
    cls = clf.predict(feat)[0]
    label = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}[cls]
    return label, prob


def main():
    df = load_dataset()
    X, y, team_gf = prepare_features(df)
    clf = train_model(X, y)

    # Example: Saudi Arabia vs Uruguay
    home = 'Saudi Arabia'
    away = 'Uruguay'
    label, prob = predict_match(clf, team_gf, home, away, neutral=False)
    print(f"\nPrediction for {home} vs {away}: {label}")
    print(f"Probabilities (Home/Draw/Away): {prob}")


if __name__ == '__main__':
    main()
