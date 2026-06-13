"""
src/scouting_engine.py

ACTIVE BREAKOUT SCOUTING ENGINE
================================
Combines two signals to find players worth watching RIGHT NOW:

1. TOPSIS Scouting Index (from strikers.py) — how good a player's CURRENT
   regularized stats look relative to their peers.

2. ML-Predicted Future Goals/90 — a Random Forest trained on
   (last season's per-90 profile + league) -> (next season's Goals/90),
   applied to each player's MOST RECENT season to project where their
   output is heading.

A "breakout" candidate is someone whose predicted future output is
meaningfully HIGHER than their current TOPSIS-implied output, especially
if they're young and/or currently under-utilized (low minutes). These are
players the underlying numbers say are about to take a step up, even if
their current reputation/minutes don't reflect it yet.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.strikers import find_breakout_strikers


# ------------------------------------------------------------------------
# 0. HELPERS
# ------------------------------------------------------------------------
def clean_conv_pct_numeric(series, fallback=12.0):
    if series.dtype == object:
        return pd.to_numeric(
            series.astype(str).str.replace('%', '', regex=False),
            errors='coerce'
        ).fillna(fallback)
    return series.fillna(fallback)


def _per90_features(df, min_minutes):
    """Add Shots_per_90 / ShT_per_90, clean Conv %, filter by minutes."""
    out = df[df['Minutes'] >= min_minutes].copy()
    out['Shots_per_90'] = (out['Shots'] / out['Minutes']) * 90
    out['ShT_per_90'] = (out['ShT'] / out['Minutes']) * 90
    out['Conv %'] = clean_conv_pct_numeric(out['Conv %'])
    return out


FEATURE_COLS = [
    'Minutes', 'Conv %', 'Goals per 90 minutes', 'xG/90',
    'Drb/90', 'Shots_per_90', 'ShT_per_90',
]


# ------------------------------------------------------------------------
# 1. TRAIN THE "NEXT SEASON G90" PREDICTOR ON HISTORICAL DATA
# ------------------------------------------------------------------------
def train_future_g90_model(league_history, min_minutes=450, min_target_minutes=900,
                            n_estimators=200, max_depth=6, random_state=42):
    """
    league_history: dict of {league_label: (s1_df, s2_df)} containing the
                     historical "year 1 -> year 2" pairs for each league.

    Returns a fitted RandomForestRegressor plus the feature column order,
    trained on a one-hot 'League' feature alongside FEATURE_COLS.
    """
    frames = []
    for league_label, (s1_df, s2_df) in league_history.items():
        s1 = _per90_features(s1_df, min_minutes)
        s1 = s1[['Player'] + FEATURE_COLS].copy()
        s1['League'] = league_label

        s2_stable = s2_df[s2_df['Minutes'] >= min_target_minutes]
        target = s2_stable[['Player', 'Goals per 90 minutes']].rename(
            columns={'Goals per 90 minutes': 'Target_G90'}
        )

        merged = pd.merge(s1, target, on='Player', how='inner').dropna()
        frames.append(merged)

    train_df = pd.concat(frames, ignore_index=True)
    train_df = pd.get_dummies(train_df, columns=['League'], drop_first=False)

    feature_cols = [c for c in train_df.columns if c not in ('Player', 'Target_G90')]
    X = train_df[feature_cols]
    y = train_df['Target_G90']

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X, y)

    return model, feature_cols, len(train_df)


# ------------------------------------------------------------------------
# 2. APPLY THE MODEL TO A PLAYER'S CURRENT SEASON TO PROJECT FUTURE G90
# ------------------------------------------------------------------------
def predict_future_g90(current_df, league_label, model, feature_cols, min_minutes=450):
    """
    current_df: a single league's most recent season dataframe (the season
                you want to scout FROM — e.g. this season's data so far).
    league_label: 'Mens' or 'Womens' (must match labels used in training).

    Returns current_df filtered to min_minutes, with a new
    'Predicted_Future_G90' column.
    """
    df = _per90_features(current_df, min_minutes)

    pred_input = df[FEATURE_COLS].copy()
    for col in feature_cols:
        if col.startswith('League_'):
            pred_input[col] = 1 if col == f'League_{league_label}' else 0
    # Ensure column order matches training
    pred_input = pred_input.reindex(columns=feature_cols, fill_value=0)

    df['Predicted_Future_G90'] = model.predict(pred_input)
    return df


# ------------------------------------------------------------------------
# 3. THE SCOUTING ENGINE: FIND ACTIVE BREAKOUT CANDIDATES
# ------------------------------------------------------------------------
def find_active_breakouts(current_df, league_label, model, feature_cols,
                           min_minutes=450, max_minutes=2000, max_age=24,
                           min_uplift=0.05, top_n=25):
    """
    Identify players whose ML-projected future Goals/90 is meaningfully
    ABOVE their current Goals/90 ("uplift"), filtered to plausible breakout
    profiles (under max_age, under max_minutes so they're not already
    entrenched starters).

    Also attaches the TOPSIS Scouting_Index_Score from strikers.py for
    cross-reference against current peer-relative quality.

    Parameters
    ----------
    min_uplift : minimum (Predicted_Future_G90 - current Goals/90) to be
                  considered a "breakout" rather than just a steady performer.

    Returns a ranked dataframe of breakout candidates.
    """
    # --- 3a. ML projection ---
    projected = predict_future_g90(current_df, league_label, model, feature_cols, min_minutes)
    projected['G90_Uplift'] = projected['Predicted_Future_G90'] - projected['Goals per 90 minutes']

    # --- 3b. TOPSIS current-quality index (only valid for strikers, min_minutes>=700 inside) ---
    try:
        topsis_df = find_breakout_strikers(current_df, min_minutes=700, max_minutes=99999)
        topsis_scores = topsis_df[['Player', 'Scouting_Index_Score']]
    except ValueError:
        # Not enough strikers passed the TOPSIS threshold — skip this signal
        topsis_scores = pd.DataFrame(columns=['Player', 'Scouting_Index_Score'])

    merged = pd.merge(projected, topsis_scores, on='Player', how='left')

    # --- 3c. Filter for breakout profile ---
    candidates = merged[
        (merged['Minutes'] <= max_minutes) &
        (merged['Age'] <= max_age) &
        (merged['G90_Uplift'] >= min_uplift)
    ].copy()

    candidates = candidates.sort_values(by='G90_Uplift', ascending=False)

    display_cols = [
        'Player', 'Club', 'Age', 'Minutes',
        'Goals per 90 minutes', 'Predicted_Future_G90', 'G90_Uplift',
        'xG/90', 'Conv %', 'Scouting_Index_Score',
    ]

    return candidates[display_cols].head(top_n).reset_index(drop=True)
