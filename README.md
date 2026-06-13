# FM26 Player Role-Fit Analytics

A tool for analyzing real performance data (not in-game attributes) from
Football Manager 26 to score how well players fit specific tactical roles,
starting with Strikers (ST (C)) in the WSL/WSL2.

## Project Structure

```
fm26-analytics/
├── data/
│   ├── raw/          # Original FM CSV exports, untouched
│   └── processed/    # Cleaned data ready for analysis
├── notebooks/        # Exploration / prototyping notebooks
├── src/
│   ├── data_loader.py  # Load and clean FM CSV exports
│   ├── roles.py        # Tactical role weight definitions
│   └── scoring.py      # Role-fit scoring logic (z-scores, weighted sums)
├── app/               # Streamlit interface (TBD)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Current Status

- [x] Data loading & cleaning (`data_loader.py`)
- [x] Role weight definitions for ST (C): Poacher, Advanced Forward, Inside Forward
- [ ] Role-fit scoring (`scoring.py`)
- [ ] Streamlit app
- [ ] Expand to other positions
