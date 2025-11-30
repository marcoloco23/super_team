"""
Model training module for Superteam.

This script loads game data from MongoDB, preprocesses it for matchup prediction,
and trains an XGBoost regression model to predict plus-minus scores.
"""

import pandas as pd
import pymongo
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

from .constants import MONGO_NAME, MONGO_PW, MONGO_DB
from .helpers import (
    flatten_performance_df,
    get_performances_by_team,
    stack_df,
    win_loss_error_rate,
)
from .logger import setup_logger

# Configuration
PLAYER_COUNT = 13  # Number of players per team
TARGET = "PLUS_MINUS"  # Target variable for prediction
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
RANDOM_STATE = 1

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
    "nthread": -1,
    "eval_metric": "rmse",
}
EARLY_STOPPING_ROUNDS = 50

# Set up logging
logger = setup_logger("model")


def get_mongo_connection():
    """
    Create MongoDB client connection.

    Returns:
        Tuple of (client, database)
    """
    client = pymongo.MongoClient(
        f"mongodb+srv://{MONGO_NAME}:{MONGO_PW}@cluster0.sfhws.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"
    )
    return client, client.superteam


def load_data(db):
    """
    Load and preprocess performance data from MongoDB.

    Args:
        db: MongoDB database instance

    Returns:
        Tuple of (player_performance_df, team_performance_df)
    """
    logger.info("Loading data from database...")

    player_performances = db.playerPerformances.find({})
    team_performances = db.teamPerformances.find({})

    player_performance_df = pd.DataFrame(list(player_performances)).set_index("_id")
    team_performance_df = pd.DataFrame(list(team_performances)).set_index("_id")

    player_performance_df = flatten_performance_df(player_performance_df)
    team_performance_df = flatten_performance_df(team_performance_df)

    return player_performance_df, team_performance_df


def preprocess_data(player_performance_df, team_performance_df, player_count=PLAYER_COUNT):
    """
    Preprocess game data into features for model training.

    Args:
        player_performance_df: DataFrame with player performance data
        team_performance_df: DataFrame with team performance data
        player_count: Number of players per team to include

    Returns:
        Tuple of (X, y) features and labels
    """
    logger.info("Preprocessing data...")

    # Find games present in both datasets
    team_game_ids = set(team_performance_df.GAME_ID)
    player_game_ids = set(player_performance_df.GAME_ID)
    game_ids = list(team_game_ids & player_game_ids)

    logger.info(f"Found {len(game_ids)} games with complete data")

    data_df_list = []
    for game_id in tqdm(game_ids, desc="Processing games"):
        game_player_performances = player_performance_df[
            player_performance_df.GAME_ID == game_id
        ].drop_duplicates()

        game_team_performances = team_performance_df[
            team_performance_df.GAME_ID == game_id
        ].drop_duplicates()

        # Split by team
        a_player, b_player = get_performances_by_team(game_player_performances, "player")
        a_team, b_team = get_performances_by_team(game_team_performances, "team")

        # Create features for team A perspective
        team_a_feature_df = pd.concat(
            [
                stack_df(
                    pd.concat(
                        [a_player[:player_count], b_player[:player_count]]
                    ).reset_index(drop=True)
                )
            ],
            axis=1,
        )
        team_a_data_df = pd.concat([team_a_feature_df, a_team[TARGET]], axis=1)

        # Create features for team B perspective
        team_b_feature_df = pd.concat(
            [
                stack_df(
                    pd.concat(
                        [b_player[:player_count], a_player[:player_count]]
                    ).reset_index(drop=True)
                )
            ],
            axis=1,
        )
        team_b_data_df = pd.concat([team_b_feature_df, b_team[TARGET]], axis=1)

        data_df_list.append(team_a_data_df)
        data_df_list.append(team_b_data_df)

    # Combine all data
    X = pd.concat(data_df_list).fillna(0).reset_index(drop=True)
    y = X.pop(TARGET)

    logger.info(f"Created {len(X)} training samples with {len(X.columns)} features")

    return X, y


def train_model(X, y):
    """
    Train XGBoost regression model.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Trained model
    """
    # Split data
    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_features, train_labels, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Training set: {len(train_features)} samples")
    logger.info(f"Validation set: {len(validation_features)} samples")
    logger.info(f"Test set: {len(test_features)} samples")

    # Create and train model
    logger.info("Training model...")
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)

    eval_set = [(validation_features, validation_labels)]
    model = model.fit(
        train_features,
        train_labels,
        eval_set=eval_set,
        verbose=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # Evaluate on test set
    predictions = model.predict(test_features)
    wler = win_loss_error_rate(predictions, test_labels)
    r2 = r2_score(test_labels, predictions)

    logger.info(f"Win Loss Accuracy: {1 - wler:.4f}")
    logger.info(f"R^2 Score: {r2:.4f}")

    return model


def main(player_count=PLAYER_COUNT):
    """
    Main training pipeline.

    Args:
        player_count: Number of players per team
    """
    # Connect to database
    client, db = get_mongo_connection()

    try:
        # Load and preprocess data
        player_performance_df, team_performance_df = load_data(db)
        X, y = preprocess_data(player_performance_df, team_performance_df, player_count)

        # Train model
        model = train_model(X, y)

        # Save model
        model_path = f"models/{player_count}_player_model.json"
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
