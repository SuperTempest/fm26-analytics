# src/data_loader.py
import pandas as pd
import re

def load_raw_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

def clean_appearances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    apps_str = df["Appearances"].astype(str)
    df["Starts"] = apps_str.str.extract(r"^(\d+)")[0].fillna(0).astype(int)
    return df

def clean_conv_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Conv %" in df.columns:
        df["Conv %"] = df["Conv %"].astype(str).str.replace("%", "", regex=False)
        df["Conv %"] = pd.to_numeric(df["Conv %"], errors="coerce").fillna(0.0)
    return df

def clean_transfer_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses FM transfer value strings (e.g. '€1.6M - €1.9M' or '€425K - €600K')
    and extracts the numeric midpoint value.
    """
    df = df.copy()
    
    def parse_value_string(val_str):
        val_str = str(val_str).replace('€', '').strip()
        if not val_str or val_str == 'nan' or val_str == '0':
            return 0.0
            
        # Extract all numbers/decimals from the range
        parts = re.findall(r'[\d\.]+[MK]?', val_str)
        numeric_values = []
        
        for part in parts:
            factor = 1.0
            if 'M' in part:
                factor = 1_000_000.0
                part = part.replace('M', '')
            elif 'K' in part:
                factor = 1_000.0
                part = part.replace('K', '')
                
            try:
                numeric_values.append(float(part) * factor)
            except ValueError:
                continue
                
        if len(numeric_values) == 2:
            return sum(numeric_values) / 2.0  # Return range midpoint
        elif len(numeric_values) == 1:
            return numeric_values[0]
        return 0.0

    df['Numeric_Value'] = df['Transfer Value'].apply(parse_value_string)
    return df

def load_and_clean(filepath: str, position: str = "ST (C)", min_minutes: int = 400) -> pd.DataFrame:
    df = load_raw_csv(filepath)
    df = clean_appearances(df)
    df = clean_conv_pct(df)
    df = clean_transfer_value(df)
    
    # Filter for active league contributors
    if position:
        df = df[df["Best Pos"] == position]
    df = df[df["Minutes"] >= min_minutes]
    
    return df.reset_index(drop=True)