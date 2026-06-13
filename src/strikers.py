# src/strikers.py
import pandas as pd
import numpy as np

def find_breakout_strikers(df: pd.DataFrame, min_minutes: int = 700, max_minutes: int = 2000) -> pd.DataFrame:
    """
    Discovers elite goalscoring breakouts using a Multi-Criteria TOPSIS Frontier Optimization 
    combined with Bayesian Regularization to eliminate small-sample-size anomalies.
    """
    # 1. Filter database for strikers with an acceptable baseline minute threshold
    df_strikers = df[(df['Best Pos'] == 'ST (C)') & (df['Minutes'] >= min_minutes)].copy()
    
    if len(df_strikers) < 5:
        raise ValueError(f"Too few players ({len(df_strikers)}) passed the min_minutes filter.")
        
    # 2. Establish Global Dataset Baselines for Regularization
    global_g90 = df_strikers['Goals per 90 minutes'].median()
    global_xg90 = df_strikers['xG/90'].median()
    
    # Safe handling of string conversion percentages
    if df_strikers['Conv %'].dtype == object:
        df_strikers['Clean_Conv'] = pd.to_numeric(
            df_strikers['Conv %'].astype(str).str.replace('%', '', regex=False), 
            errors='coerce'
        ).fillna(12.0)
    else:
        df_strikers['Clean_Conv'] = df_strikers['Conv %'].fillna(12.0)
        
    global_conv = df_strikers['Clean_Conv'].median()
    
    # 3. APPLY BAYESIAN-STYLE SHRINKAGE (Penalize low-minute hot streaks)
    CONFIDENCE_MINUTES = 2200.0
    reg_g90, reg_xg90, reg_conv = [], [], []
    
    for idx, row in df_strikers.iterrows():
        trust_factor = min(1.0, row['Minutes'] / CONFIDENCE_MINUTES)
        
        reg_g90.append((row['Goals per 90 minutes'] * trust_factor) + (global_g90 * (1 - trust_factor)))
        reg_xg90.append((row['xG/90'] * trust_factor) + (global_xg90 * (1 - trust_factor)))
        reg_conv.append((row['Clean_Conv'] * trust_factor) + (global_conv * (1 - trust_factor)))
        
    df_strikers['Reg_G90'] = reg_g90
    df_strikers['Reg_xG90'] = reg_xg90
    df_strikers['Reg_Conv'] = reg_conv
    
    # 4. PREPARE MATHEMATICAL ARRAYS FOR THE TOPSIS ALGORITHM
    features = ['Reg_G90', 'Reg_xG90', 'Reg_Conv']
    X = df_strikers[features].values
    
    # Vector Normalization (Ensures metrics are on the same geometric scale)
    norm_denominator = np.sqrt(np.sum(X**2, axis=0))
    norm_denominator[norm_denominator == 0] = 1e-6
    norm_X = X / norm_denominator
    
    # Apply Importance Weights: 45% Goals/90, 35% xG/90, 20% Conversion Efficiency
    weights = np.array([0.45, 0.35, 0.20])
    weighted_X = norm_X * weights
    
    # 5. DETERMINE THE IDEAL BEST AND IDEAL WORST BOUNDARIES
    ideal_best = np.max(weighted_X, axis=0)
    ideal_worst = np.min(weighted_X, axis=0)
    
    # 6. CALCULATE EUCLIDEAN DISTANCES TO BOTH FRONTIERS
    d_best = np.sqrt(np.sum((weighted_X - ideal_best)**2, axis=1))
    d_worst = np.sqrt(np.sum((weighted_X - ideal_worst)**2, axis=1))
    
    # Compute relative closeness to the ideal solution (TOPSIS Score)
    scores = d_worst / (d_best + d_worst + 1e-9)
    df_strikers['Scouting_Index_Score'] = scores * 100
    
    # 7. FILTER FOR BREAKOUT TARGETS (Under-utilized or developmental minute profiles)
    breakout_candidates = df_strikers[df_strikers['Minutes'] <= max_minutes].copy()
    
    # Sort from absolute highest performance index downwards
    breakout_candidates = breakout_candidates.sort_values(by='Scouting_Index_Score', ascending=False)
    
    display_cols = [
        'Player', 'Club', 'Age', 'Minutes', 
        'Goals per 90 minutes', 'xG/90', 'Conv %', 'Scouting_Index_Score'
    ]
    
    return breakout_candidates[display_cols].reset_index(drop=True)