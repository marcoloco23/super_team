"""
Unit tests for the simulation module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from superteam.simulation import (
    simulate_arbitrary_matchup,
    simulate_nba_matchup,
    evaluate_team,
    get_super_team,
)


class TestSimulateArbitraryMatchup:
    """Tests for the simulate_arbitrary_matchup function."""

    def test_returns_predictions(self, sample_average_performances, mock_xgb_model):
        """Test that function returns model predictions."""
        team_a_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        team_b_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        result = simulate_arbitrary_matchup(
            team_a_ids,
            team_b_ids,
            sample_average_performances,
            mock_xgb_model,
            team_size=13,
        )

        # Should return predictions array
        assert result is not None
        assert len(result) == 2  # Two predictions (one per team)

    def test_model_called_with_features(self, sample_average_performances, mock_xgb_model):
        """Test that model.predict is called."""
        team_a_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        team_b_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        simulate_arbitrary_matchup(
            team_a_ids,
            team_b_ids,
            sample_average_performances,
            mock_xgb_model,
            team_size=13,
        )

        # Model predict should have been called
        mock_xgb_model.predict.assert_called_once()


class TestSimulateNbaMatchup:
    """Tests for the simulate_nba_matchup function."""

    def test_returns_predictions(self, sample_average_performances, mock_xgb_model):
        """Test that function returns predictions for NBA teams."""
        result = simulate_nba_matchup(
            "LAL",
            "BOS",
            sample_average_performances,
            mock_xgb_model,
            team_size=13,
        )

        assert result is not None
        assert len(result) == 2

    def test_handles_injured_players(self, sample_average_performances, mock_xgb_model):
        """Test that injured players are excluded."""
        result = simulate_nba_matchup(
            "LAL",
            "BOS",
            sample_average_performances,
            mock_xgb_model,
            team_A_injured_player_ids=[1, 2],
            team_size=13,
        )

        # Should still return predictions
        assert result is not None
        assert len(result) == 2

    def test_with_different_team_sizes(self, sample_average_performances, mock_xgb_model):
        """Test with different team sizes."""
        for team_size in [5, 8, 10]:
            result = simulate_nba_matchup(
                "LAL",
                "BOS",
                sample_average_performances,
                mock_xgb_model,
                team_size=team_size,
            )
            assert result is not None


class TestEvaluateTeam:
    """Tests for the evaluate_team function."""

    def test_returns_win_rate(self, sample_average_performances, mock_xgb_model):
        """Test that function returns a win rate."""
        team_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        # Mock to return alternating wins
        mock_xgb_model.predict.side_effect = [
            np.array([5.0, -5.0]),  # Win
            np.array([-5.0, 5.0]),  # Loss
        ] * 50

        result = evaluate_team(
            team_ids,
            sample_average_performances,
            mock_xgb_model,
            team_size=13,
            iterations=10,
        )

        # Should return a value between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_perfect_team(self, sample_average_performances, mock_xgb_model):
        """Test with a team that always wins."""
        team_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        # Always win
        mock_xgb_model.predict.return_value = np.array([10.0, -10.0])

        result = evaluate_team(
            team_ids,
            sample_average_performances,
            mock_xgb_model,
            team_size=13,
            iterations=5,
        )

        assert result == 1.0


class TestGetSuperTeam:
    """Tests for the get_super_team function."""

    def test_returns_team_ids(self, sample_average_performances, mock_xgb_model):
        """Test that function returns a list of player IDs."""
        # Mock score_df
        with patch("superteam.simulation.get_score_df") as mock_score:
            score_df = sample_average_performances.copy()
            score_df["SCORE"] = np.random.uniform(0, 1, len(score_df))
            mock_score.return_value = score_df

            with patch("superteam.simulation.get_salary_cap") as mock_cap:
                mock_cap.return_value = 10.0

                result = get_super_team(
                    sample_average_performances,
                    mock_xgb_model,
                    team_size=5,
                    iterations=3,
                    salary_cap=False,
                )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 5

    def test_respects_team_size(self, sample_average_performances, mock_xgb_model):
        """Test that returned team has correct size."""
        with patch("superteam.simulation.get_score_df") as mock_score:
            score_df = sample_average_performances.copy()
            score_df["SCORE"] = np.random.uniform(0, 1, len(score_df))
            mock_score.return_value = score_df

            with patch("superteam.simulation.get_salary_cap") as mock_cap:
                mock_cap.return_value = 10.0

                for size in [5, 8, 10]:
                    result = get_super_team(
                        sample_average_performances,
                        mock_xgb_model,
                        team_size=size,
                        iterations=2,
                        salary_cap=False,
                    )
                    assert len(result) == size


class TestSimulationEdgeCases:
    """Test edge cases in simulation functions."""

    def test_empty_predictions_handling(self, sample_average_performances):
        """Test handling when model returns empty predictions."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([])

        # This should not crash
        team_a_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        team_b_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

        result = simulate_arbitrary_matchup(
            team_a_ids,
            team_b_ids,
            sample_average_performances,
            mock_model,
            team_size=13,
        )

        assert result is not None

    def test_single_player_team(self, sample_average_performances, mock_xgb_model):
        """Test simulation with single player teams."""
        result = simulate_nba_matchup(
            "LAL",
            "BOS",
            sample_average_performances,
            mock_xgb_model,
            team_size=1,
        )

        assert result is not None
        assert len(result) == 2
