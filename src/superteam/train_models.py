"""
Model training module for Superteam.

Trains XGBoost models using the same feature processing as simulation.py.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from typing import Tuple, Dict, Any
import json
from datetime import datetime

try:
    from .helpers import stack_df, win_loss_error_rate
    from .logger import setup_logger
except ImportError:
    from superteam.helpers import stack_df, win_loss_error_rate
    from superteam.logger import setup_logger

logger = setup_logger("train_models")

# Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
PLAYER_DATA_FILE = "player_data.csv"
TARGET = "PLUS_MINUS"
START_COLUMN = "PCT_FGA_2PT"  # Features start here (same as simulation.py)

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "booster": "gbtree",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 4,
    "min_child_weight": 4,
    "gamma": 0.6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": -1,
}


def load_data() -> pd.DataFrame:
    """Load and prepare player data."""
    filepath = os.path.join(DATA_DIR, PLAYER_DATA_FILE)
    logger.info(f"Loading {filepath}...")

    df = pd.read_csv(filepath, index_col=0, low_memory=False)

    # Convert MIN to numeric
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)

    logger.info(f"Loaded {len(df)} records")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns starting from PCT_FGA_2PT, numeric only."""
    start_idx = df.columns.get_loc(START_COLUMN)
    all_cols = df.columns[start_idx:].tolist()

    # Keep only numeric columns
    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Remove target from features
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    return numeric_cols


def prepare_game_data(game_df: pd.DataFrame, player_count: int, feature_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features for one game (both teams)."""
    teams = game_df['TEAM_ABBREVIATION'].unique()
    if len(teams) != 2:
        return None, None

    team_a = game_df[game_df['TEAM_ABBREVIATION'] == teams[0]].sort_values('MIN', ascending=False)
    team_b = game_df[game_df['TEAM_ABBREVIATION'] == teams[1]].sort_values('MIN', ascending=False)

    # Get top players' features
    a_features = team_a[feature_cols].head(player_count).reset_index(drop=True)
    b_features = team_b[feature_cols].head(player_count).reset_index(drop=True)

    # Pad if needed
    for _ in range(player_count - len(a_features)):
        a_features = pd.concat([a_features, pd.DataFrame([{c: 0 for c in feature_cols}])], ignore_index=True)
    for _ in range(player_count - len(b_features)):
        b_features = pd.concat([b_features, pd.DataFrame([{c: 0 for c in feature_cols}])], ignore_index=True)

    # Get targets
    target_a = team_a[TARGET].iloc[0] if len(team_a) > 0 else 0
    target_b = team_b[TARGET].iloc[0] if len(team_b) > 0 else 0

    # Stack: team A vs team B
    combined_a = pd.concat([a_features, b_features]).reset_index(drop=True)
    combined_b = pd.concat([b_features, a_features]).reset_index(drop=True)

    row_a = stack_df(combined_a)
    row_b = stack_df(combined_b)

    row_a[TARGET] = target_a
    row_b[TARGET] = target_b

    return row_a, row_b


def build_training_data(df: pd.DataFrame, player_count: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Build training dataset from game data."""
    logger.info(f"Building training data for {player_count}-player model...")

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")

    game_ids = df['GAME_ID'].unique()
    logger.info(f"Processing {len(game_ids)} games...")

    rows = []
    for game_id in tqdm(game_ids, desc="Games"):
        game_df = df[df['GAME_ID'] == game_id]
        row_a, row_b = prepare_game_data(game_df, player_count, feature_cols)
        if row_a is not None:
            rows.append(row_a)
            rows.append(row_b)

    if not rows:
        raise ValueError("No valid games found")

    data = pd.concat(rows, ignore_index=True).fillna(0)
    y = data.pop(TARGET)
    X = data

    logger.info(f"Created {len(X)} samples with {len(X.columns)} features")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.XGBRegressor, Dict]:
    """Train XGBoost model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)

    metrics = {
        'r2': round(r2_score(y_test, y_pred), 4),
        'mae': round(mean_absolute_error(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'win_loss_acc': round(1 - win_loss_error_rate(y_pred, y_test), 4),
        'features': len(X.columns),
        'samples': len(X),
    }

    logger.info(f"R²={metrics['r2']}, WL Acc={metrics['win_loss_acc']}, RMSE={metrics['rmse']}")
    return model, metrics


def train_all_models(player_counts=[1, 5, 8, 10, 13]):
    """Train all models."""
    df = load_data()

    # Filter to rows with valid feature data (PCT_FGA_2PT not null)
    df = df[df['PCT_FGA_2PT'].notna()]
    logger.info(f"Using {len(df)} records with valid features ({df['GAME_ID'].nunique()} games)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for pc in player_counts:
        logger.info(f"\n{'='*40}\nTraining {pc}-player model\n{'='*40}")

        try:
            X, y = build_training_data(df, pc)
            model, metrics = train_model(X, y)

            path = os.path.join(MODELS_DIR, f"{pc}_player_model.json")
            model.save_model(path)
            logger.info(f"Saved to {path}")

            results[pc] = metrics
        except Exception as e:
            logger.error(f"Failed: {e}")

    # Save report
    report = {'timestamp': datetime.now().isoformat(), 'models': results}
    with open(os.path.join(MODELS_DIR, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    return results


if __name__ == "__main__":
    train_all_models()
