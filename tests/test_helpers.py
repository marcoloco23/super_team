"""
Unit tests for the helpers module.
"""

import pytest
import pandas as pd
import numpy as np
from superteam.helpers import (
    stack_df,
    win_loss_error_rate,
    make_data_relative,
    get_team_feature_df,
    combine_team_games,
)


class TestStackDf:
    """Tests for the stack_df function."""

    def test_stack_df_basic(self):
        """Test basic stacking of a DataFrame."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        })
        result = stack_df(df)

        # Should have 1 row
        assert len(result) == 1

        # Should have columns like A_1, B_1, A_2, B_2, etc.
        assert "A_1" in result.columns
        assert "B_1" in result.columns
        assert "A_2" in result.columns
        assert "B_2" in result.columns

    def test_stack_df_values(self):
        """Test that stacked values are correct."""
        df = pd.DataFrame({
            "A": [10, 20],
            "B": [30, 40],
        })
        result = stack_df(df)

        assert result["A_1"].values[0] == 10
        assert result["A_2"].values[0] == 20
        assert result["B_1"].values[0] == 30
        assert result["B_2"].values[0] == 40

    def test_stack_df_preserves_data(self):
        """Test that stacking preserves all original data."""
        df = pd.DataFrame({
            "col1": [1.5, 2.5, 3.5],
            "col2": [4.5, 5.5, 6.5],
        })
        result = stack_df(df)

        # Total values should be rows * cols
        assert len(result.columns) == 6  # 3 rows * 2 cols


class TestWinLossErrorRate:
    """Tests for the win_loss_error_rate function."""

    def test_perfect_predictions(self):
        """Test error rate with perfect predictions."""
        predictions = np.array([5.0, -3.0, 2.0, -1.0])
        labels = pd.Series([5.0, -3.0, 2.0, -1.0])

        error_rate = win_loss_error_rate(predictions, labels)
        assert error_rate == 0.0

    def test_all_wrong_predictions(self):
        """Test error rate when all predictions are wrong."""
        predictions = np.array([5.0, 3.0, 2.0, 1.0])  # All positive
        labels = pd.Series([-5.0, -3.0, -2.0, -1.0])  # All negative

        error_rate = win_loss_error_rate(predictions, labels)
        assert error_rate == 1.0

    def test_half_correct_predictions(self):
        """Test error rate with 50% correct predictions."""
        predictions = np.array([5.0, -3.0, 2.0, -1.0])
        labels = pd.Series([5.0, 3.0, 2.0, 1.0])  # 2 correct, 2 wrong

        error_rate = win_loss_error_rate(predictions, labels)
        assert error_rate == 0.5

    def test_zero_values(self):
        """Test handling of zero values."""
        predictions = np.array([0.0, 5.0, -3.0])
        labels = pd.Series([0.0, 5.0, -3.0])

        # 0 is treated as loss (not > 0)
        error_rate = win_loss_error_rate(predictions, labels)
        assert error_rate == 0.0


class TestMakeDataRelative:
    """Tests for the make_data_relative function."""

    def test_basic_relative(self):
        """Test basic relative calculation."""
        df = pd.DataFrame({
            "stat1": [10.0, 5.0],
            "stat2": [20.0, 30.0],
        })
        result = make_data_relative(df)

        # First row should be diff (row0 - row1)
        assert result.iloc[0]["stat1"] == 5.0  # 10 - 5
        assert result.iloc[0]["stat2"] == -10.0  # 20 - 30

        # Second row should be diff (row1 - row0)
        assert result.iloc[1]["stat1"] == -5.0  # 5 - 10
        assert result.iloc[1]["stat2"] == 10.0  # 30 - 20

    def test_symmetric_differences(self):
        """Test that differences are symmetric."""
        df = pd.DataFrame({
            "points": [100.0, 95.0],
            "rebounds": [50.0, 45.0],
        })
        result = make_data_relative(df)

        # Row 0 + Row 1 should equal 0 for each column
        for col in result.columns:
            assert result[col].sum() == 0.0


class TestGetTeamFeatureDf:
    """Tests for the get_team_feature_df function."""

    def test_combines_team_features(self):
        """Test that team features are properly combined."""
        team_a = pd.DataFrame({
            "stat1": [1.0, 2.0],
            "stat2": [3.0, 4.0],
        })
        team_b = pd.DataFrame({
            "stat1": [5.0, 6.0],
            "stat2": [7.0, 8.0],
        })

        result = get_team_feature_df(team_a, team_b)

        # Should have 1 row
        assert len(result) == 1

        # Should have features from both teams
        assert len(result.columns) == 8  # 2 stats * 4 rows (2 per team)

    def test_feature_df_structure(self):
        """Test the structure of the combined feature DataFrame."""
        team_a = pd.DataFrame({
            "points": [100.0],
            "assists": [25.0],
        })
        team_b = pd.DataFrame({
            "points": [95.0],
            "assists": [20.0],
        })

        result = get_team_feature_df(team_a, team_b)

        # Should be a single row
        assert result.shape[0] == 1


class TestCombineTeamGames:
    """Tests for the combine_team_games function."""

    def test_home_filter(self):
        """Test filtering for home games."""
        df = pd.DataFrame({
            "SEASON_ID": ["2023", "2023"],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2023-01-01", "2023-01-01"],
            "TEAM_ID": [1, 2],
            "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
            "MATCHUP_A": ["LAL vs. BOS", "BOS @ LAL"],
            "WL": ["W", "L"],
            "WL_A": ["W", "L"],
        })

        result = combine_team_games(df, keep_method="home")

        # Should only have home team perspective
        assert len(result) > 0
        assert all(result["MATCHUP_A"].str.contains(" vs. "))

    def test_invalid_keep_method(self):
        """Test that invalid keep_method raises ValueError."""
        df = pd.DataFrame({
            "SEASON_ID": ["2023"],
            "GAME_ID": ["001"],
            "GAME_DATE": ["2023-01-01"],
            "TEAM_ID": [1],
        })

        with pytest.raises(ValueError, match="Invalid keep_method"):
            combine_team_games(df, keep_method="invalid")

    def test_winner_filter(self):
        """Test filtering for winners."""
        # Need to create proper data that can be self-joined
        df = pd.DataFrame({
            "SEASON_ID": ["2023", "2023"],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2023-01-01", "2023-01-01"],
            "TEAM_ID": [1, 2],
            "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
            "WL": ["W", "L"],
        })

        result = combine_team_games(df, keep_method="winner")

        # Should only have winners (the joined data will have WL_A suffix)
        if len(result) > 0:
            assert all(result["WL_A"] == "W")
