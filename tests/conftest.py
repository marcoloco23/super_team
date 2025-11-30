"""
Pytest configuration and fixtures for Superteam tests.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_player_data():
    """Create sample player performance data for testing."""
    np.random.seed(42)
    n_players = 30

    data = {
        "PLAYER_ID": list(range(1, n_players + 1)),
        "PLAYER_NAME": [f"Player_{i}" for i in range(1, n_players + 1)],
        "TEAM_ABBREVIATION": ["LAL"] * 15 + ["BOS"] * 15,
        "TEAM_ID": [1] * 15 + [2] * 15,
        "GAME_ID": ["0021900001"] * n_players,
        "MIN": np.random.uniform(10, 40, n_players),
        "PCT_FGA_2PT": np.random.uniform(0.3, 0.7, n_players),
        "PCT_AST_2PM": np.random.uniform(0.1, 0.5, n_players),
        "PCT_PTS_2PT": np.random.uniform(0.2, 0.6, n_players),
        "AST_PCT": np.random.uniform(0.05, 0.3, n_players),
        "FG3_PCT": np.random.uniform(0.25, 0.45, n_players),
        "FG_PCT": np.random.uniform(0.4, 0.55, n_players),
        "FT_PCT": np.random.uniform(0.6, 0.9, n_players),
        "PLUS_MINUS": np.random.uniform(-15, 15, n_players),
        "PTS": np.random.randint(0, 35, n_players),
        "AST": np.random.randint(0, 15, n_players),
        "REB": np.random.randint(0, 15, n_players),
        "STL": np.random.randint(0, 5, n_players),
        "BLK": np.random.randint(0, 5, n_players),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_average_performances(sample_player_data):
    """Create sample average performances DataFrame."""
    return sample_player_data.copy()


@pytest.fixture
def sample_team_data():
    """Create sample team performance data for testing."""
    np.random.seed(42)

    data = {
        "TEAM_ID": [1, 2],
        "TEAM_ABBREVIATION": ["LAL", "BOS"],
        "GAME_ID": ["0021900001", "0021900001"],
        "MIN": [240.0, 240.0],
        "PCT_FGA_2PT": [0.5, 0.48],
        "PCT_AST_2PM": [0.3, 0.32],
        "PCT_PTS_2PT": [0.4, 0.42],
        "PLUS_MINUS": [5, -5],
        "PTS": [110, 105],
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_xgb_model():
    """Create a mock XGBoost model for testing."""
    model = MagicMock()
    # Return different predictions for two teams
    model.predict.return_value = np.array([5.0, -5.0])
    return model


@pytest.fixture
def sample_box_score_df():
    """Create a sample box score DataFrame."""
    data = {
        "PLAYER_ID": [1, 2, 3, 4, 5],
        "TEAM_ID": [1, 1, 1, 2, 2],
        "TEAM_ABBREVIATION": ["LAL", "LAL", "LAL", "BOS", "BOS"],
        "MIN": ["32:15", "28:30", "25:45", "35:00", "30:20"],
        "PTS": [25, 18, 12, 30, 15],
        "AST": [8, 5, 3, 6, 4],
        "REB": [10, 4, 8, 5, 12],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_game_performance_df():
    """Create sample game performance data with team abbreviations."""
    np.random.seed(42)
    n_players = 26  # 13 per team

    data = {
        "PLAYER_ID": list(range(1, n_players + 1)),
        "PLAYER_NAME": [f"Player_{i}" for i in range(1, n_players + 1)],
        "TEAM_ABBREVIATION": ["LAL"] * 13 + ["BOS"] * 13,
        "TEAM_ID": [1] * 13 + [2] * 13,
        "GAME_ID": ["0021900001"] * n_players,
        "GAME_DATE": ["2023-01-01"] * n_players,
        "START_POSITION": ["G", "G", "F", "F", "C"] * 5 + ["G"],
        "NICKNAME": [f"Nick_{i}" for i in range(1, n_players + 1)],
        "TEAM_CITY": ["Los Angeles"] * 13 + ["Boston"] * 13,
        "MIN": np.random.uniform(10, 40, n_players),
        "PCT_FGA_2PT": np.random.uniform(0.3, 0.7, n_players),
        "PCT_AST_2PM": np.random.uniform(0.1, 0.5, n_players),
        "PLUS_MINUS": np.random.uniform(-15, 15, n_players),
    }

    return pd.DataFrame(data)


@pytest.fixture
def teams_df():
    """Create a sample teams DataFrame."""
    data = {
        "id": [1610612747, 1610612738],
        "full_name": ["Los Angeles Lakers", "Boston Celtics"],
        "abbreviation": ["LAL", "BOS"],
        "nickname": ["Lakers", "Celtics"],
        "city": ["Los Angeles", "Boston"],
        "state": ["California", "Massachusetts"],
    }
    return pd.DataFrame(data)
