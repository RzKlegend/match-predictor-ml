"""Minimal ML-based match predictor.

Interactive CLI: asks for two teams, trains a small RandomForest from available data
and outputs a predicted winner plus an ELO-based check computed from recent matches.

Place `football_data.csv` in the project root to use real data. The script falls back
to a small synthetic sample if the file is missing.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from difflib import get_close_matches


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
            hs = int(rng.poisson(1.5))
            as_ = int(rng.poisson(1.2))
            rows.append({'date': '2020-01-01', 'home_team': h, 'away_team': a, 'home_score': hs, 'away_score': as_, 'neutral': False})
        return pd.DataFrame(rows)


def prepare_features(df):
    # Drop matches without scores
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    # Target: 0=home win,1=draw,2=away win
    df['result'] = df.apply(lambda r: 0 if r['home_score']>r['away_score'] else (1 if r['home_score']==r['away_score'] else 2), axis=1)

    # Team-level aggregates (goals for average)
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
    if len(np.unique(y)) < 2:
        print("Not enough classes to train. Using fallback random predictor.")
        class Dummy:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
            def predict_proba(self, X):
                return np.tile([1.0, 0.0, 0.0], (len(X), 1))
        return Dummy()

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


def calculate_team_elo(df, team_name, num_matches=50):
    """Compute a simple ELO-like rating from recent completed matches."""
    home_matches = df[(df['home_team'] == team_name) & (df['home_score'].notna())].tail(num_matches)
    away_matches = df[(df['away_team'] == team_name) & (df['away_score'].notna())].tail(num_matches)
    elo = 1600
    for _, row in pd.concat([home_matches, away_matches]).iterrows():
        if row['home_team'] == team_name:
            s_for = int(row['home_score']); s_against = int(row['away_score'])
        else:
            s_for = int(row['away_score']); s_against = int(row['home_score'])
        if s_for > s_against:
            elo += 16
        elif s_for < s_against:
            elo -= 16
        # draws -> no change
    return elo


def ask_country(name_prompt, teams):
    t = input(name_prompt).strip()
    if t in teams:
        return t
    # try fuzzy matching
    matches = get_close_matches(t, teams, n=3, cutoff=0.6)
    if matches:
        print(f"Team '{t}' not found. Did you mean: {', '.join(matches)} ?")
        choice = input(f"Enter exact team name from suggestions or press Enter to cancel: ").strip()
        if choice in teams:
            return choice
    print(f"Country '{t}' not found and no selection made.")
    return None


def main():
    df = load_dataset()
    X, y, team_gf = prepare_features(df)
    clf = train_model(X, y)

    teams = sorted(list(team_gf.keys()))
    print("\nEnter two countries to predict a match outcome. Example: Saudi Arabia")
    home = ask_country('Home country: ', teams)
    if not home:
        return
    away = ask_country('Away country: ', teams)
    if not away:
        return
    neutral_in = input('Is the match on neutral ground? (y/N): ').strip().lower()
    neutral = neutral_in == 'y'

    label, prob = predict_match(clf, team_gf, home, away, neutral=neutral)
    print(f"\nModel prediction for {home} vs {away}: {label}")
    print(f"Probabilities (Home/Draw/Away): {prob}")

    # ELO-based fallback/check
    try:
        elo_h = calculate_team_elo(df, home)
        elo_a = calculate_team_elo(df, away)
        if elo_h > elo_a:
            elo_winner = home
        elif elo_a > elo_h:
            elo_winner = away
        else:
            elo_winner = 'Draw'
        print(f"\nELO check — {home}: {elo_h}, {away}: {elo_a} -> {elo_winner}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
