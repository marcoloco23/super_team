"""
Unit tests for the Pydantic models.
"""

import pytest
from superteam.models import (
    PlayerPercentages,
    TeamPercentages,
    Ratings,
    Misc,
    AbsoluteStatistics,
    PlayerPerformance,
    TeamPerformance,
)


class TestPlayerPercentages:
    """Tests for the PlayerPercentages model."""

    def test_valid_percentages(self):
        """Test creating valid player percentages."""
        data = {
            "PCT_FGA_2PT": 0.5,
            "PCT_AST_2PM": 0.3,
            "PCT_PTS_2PT": 0.4,
            "AST_PCT": 0.15,
            "PCT_FG3M": 0.35,
            "PCT_BLKA": 0.1,
            "PCT_BLK": 0.05,
            "FG3_PCT": 0.38,
            "PCT_PTS": 0.12,
            "PCT_FGM": 0.08,
            "PCT_REB": 0.07,
            "PCT_FGA": 0.09,
            "E_USG_PCT": 0.2,
            "REB_PCT": 0.1,
            "PCT_PTS_OFF_TOV": 0.05,
            "PCT_DREB": 0.12,
            "OPP_OREB_PCT": 0.25,
            "PCT_UAST_3PM": 0.4,
            "PCT_TOV": 0.1,
            "DREB_PCT": 0.15,
            "PCT_FTM": 0.08,
            "OPP_TOV_PCT": 0.12,
            "PCT_UAST_2PM": 0.35,
            "PCT_AST_3PM": 0.3,
            "USG_PCT": 0.22,
            "PCT_AST": 0.18,
            "FG_PCT": 0.48,
            "EFG_PCT": 0.52,
            "TS_PCT": 0.55,
            "PCT_OREB": 0.08,
            "PCT_PTS_2PT_MR": 0.15,
            "PCT_PF": 0.1,
            "FT_PCT": 0.82,
            "PCT_PTS_PAINT": 0.3,
            "PCT_PTS_FT": 0.15,
            "PCT_PFD": 0.08,
            "PCT_FGA_3PT": 0.35,
            "OPP_EFG_PCT": 0.5,
            "CFG_PCT": 0.45,
            "TM_TOV_PCT": 0.12,
            "PCT_UAST_FGM": 0.4,
            "PCT_PTS_3PT": 0.25,
            "OREB_PCT": 0.1,
            "PCT_PTS_FB": 0.08,
            "PCT_AST_FGM": 0.35,
            "UFG_PCT": 0.48,
            "PCT_FG3A": 0.3,
            "PCT_STL": 0.05,
            "DFG_PCT": 0.45,
        }

        percentages = PlayerPercentages(**data)
        assert percentages.FG_PCT == 0.48
        assert percentages.FT_PCT == 0.82


class TestRatings:
    """Tests for the Ratings model."""

    def test_valid_ratings(self):
        """Test creating valid ratings."""
        data = {
            "E_OFF_RATING": 115.5,
            "OFF_RATING": 112.3,
            "E_NET_RATING": 5.2,
            "E_DEF_RATING": 110.3,
            "NET_RATING": 4.5,
            "DEF_RATING": 107.8,
        }

        ratings = Ratings(**data)
        assert ratings.OFF_RATING == 112.3
        assert ratings.DEF_RATING == 107.8


class TestMisc:
    """Tests for the Misc model."""

    def test_valid_misc(self):
        """Test creating valid misc stats."""
        data = {
            "E_PACE": 100.5,
            "AST_RATIO": 18.5,
            "DIST": 2.8,
            "AST_TOV": 2.1,
            "FTA_RATE": 0.25,
            "OPP_FTA_RATE": 0.22,
            "MIN": 32.5,
            "PACE_PER40": 100.0,
            "PACE": 98.5,
            "PIE": 0.12,
        }

        misc = Misc(**data)
        assert misc.MIN == 32.5
        assert misc.PIE == 0.12


class TestAbsoluteStatistics:
    """Tests for the AbsoluteStatistics model."""

    def test_valid_absolute_stats(self):
        """Test creating valid absolute statistics."""
        data = {
            "OREB": 2,
            "AST": 8,
            "REB": 10,
            "DFGA": 15,
            "SAST": 3,
            "OPP_PTS_2ND_CHANCE": 10,
            "PFD": 4,
            "TO": 3,
            "FG3A": 8,
            "STL": 2,
            "POSS": 95,
            "PASS": 45,
            "UFGM": 5,
            "FG3M": 3,
            "PTS": 25,
            "UFGA": 12,
            "DRBC": 8,
            "OPP_PTS_PAINT": 40,
            "FTM": 5,
            "ORBC": 2,
            "BLKA": 1,
            "PTS_FB": 8,
            "CFGA": 10,
            "PTS_PAINT": 12,
            "TCHS": 60,
            "CFGM": 5,
            "PLUS_MINUS": 7,
            "DFGM": 6,
            "OPP_PTS_OFF_TOV": 12,
            "PTS_OFF_TOV": 15,
            "FGA": 18,
            "FTA": 6,
            "PTS_2ND_CHANCE": 8,
            "FGM": 9,
            "PF": 3,
            "DREB": 8,
            "BLK": 1,
            "RBC": 10,
            "OPP_PTS_FB": 10,
            "FTAST": 2,
        }

        stats = AbsoluteStatistics(**data)
        assert stats.PTS == 25
        assert stats.AST == 8
        assert stats.PLUS_MINUS == 7


class TestPlayerPerformance:
    """Tests for the PlayerPerformance model."""

    def test_valid_player_performance(self):
        """Test creating a valid player performance."""
        # Create nested models
        percentages_data = {
            "PCT_FGA_2PT": 0.5, "PCT_AST_2PM": 0.3, "PCT_PTS_2PT": 0.4,
            "AST_PCT": 0.15, "PCT_FG3M": 0.35, "PCT_BLKA": 0.1,
            "PCT_BLK": 0.05, "FG3_PCT": 0.38, "PCT_PTS": 0.12,
            "PCT_FGM": 0.08, "PCT_REB": 0.07, "PCT_FGA": 0.09,
            "E_USG_PCT": 0.2, "REB_PCT": 0.1, "PCT_PTS_OFF_TOV": 0.05,
            "PCT_DREB": 0.12, "OPP_OREB_PCT": 0.25, "PCT_UAST_3PM": 0.4,
            "PCT_TOV": 0.1, "DREB_PCT": 0.15, "PCT_FTM": 0.08,
            "OPP_TOV_PCT": 0.12, "PCT_UAST_2PM": 0.35, "PCT_AST_3PM": 0.3,
            "USG_PCT": 0.22, "PCT_AST": 0.18, "FG_PCT": 0.48,
            "EFG_PCT": 0.52, "TS_PCT": 0.55, "PCT_OREB": 0.08,
            "PCT_PTS_2PT_MR": 0.15, "PCT_PF": 0.1, "FT_PCT": 0.82,
            "PCT_PTS_PAINT": 0.3, "PCT_PTS_FT": 0.15, "PCT_PFD": 0.08,
            "PCT_FGA_3PT": 0.35, "OPP_EFG_PCT": 0.5, "CFG_PCT": 0.45,
            "TM_TOV_PCT": 0.12, "PCT_UAST_FGM": 0.4, "PCT_PTS_3PT": 0.25,
            "OREB_PCT": 0.1, "PCT_PTS_FB": 0.08, "PCT_AST_FGM": 0.35,
            "UFG_PCT": 0.48, "PCT_FG3A": 0.3, "PCT_STL": 0.05, "DFG_PCT": 0.45,
        }

        abs_stats_data = {
            "OREB": 2, "AST": 8, "REB": 10, "DFGA": 15, "SAST": 3,
            "OPP_PTS_2ND_CHANCE": 10, "PFD": 4, "TO": 3, "FG3A": 8,
            "STL": 2, "POSS": 95, "PASS": 45, "UFGM": 5, "FG3M": 3,
            "PTS": 25, "UFGA": 12, "DRBC": 8, "OPP_PTS_PAINT": 40,
            "FTM": 5, "ORBC": 2, "BLKA": 1, "PTS_FB": 8, "CFGA": 10,
            "PTS_PAINT": 12, "TCHS": 60, "CFGM": 5, "PLUS_MINUS": 7,
            "DFGM": 6, "OPP_PTS_OFF_TOV": 12, "PTS_OFF_TOV": 15,
            "FGA": 18, "FTA": 6, "PTS_2ND_CHANCE": 8, "FGM": 9,
            "PF": 3, "DREB": 8, "BLK": 1, "RBC": 10, "OPP_PTS_FB": 10, "FTAST": 2,
        }

        ratings_data = {
            "E_OFF_RATING": 115.5, "OFF_RATING": 112.3, "E_NET_RATING": 5.2,
            "E_DEF_RATING": 110.3, "NET_RATING": 4.5, "DEF_RATING": 107.8,
        }

        misc_data = {
            "E_PACE": 100.5, "AST_RATIO": 18.5, "DIST": 2.8, "AST_TOV": 2.1,
            "FTA_RATE": 0.25, "OPP_FTA_RATE": 0.22, "MIN": 32.5,
            "PACE_PER40": 100.0, "PACE": 98.5, "PIE": 0.12,
        }

        performance = PlayerPerformance(
            GAME_ID="0021900001",
            GAME_DATE="2023-01-15",
            TEAM_ID=1610612747,
            TEAM_ABBREVIATION="LAL",
            TEAM_CITY="Los Angeles",
            PLAYER_ID=2544,
            PLAYER_NAME="LeBron James",
            NICKNAME="King",
            START_POSITION="F",
            PERCENTAGES=PlayerPercentages(**percentages_data),
            ABSOLUTE_STATISTICS=AbsoluteStatistics(**abs_stats_data),
            RATINGS=Ratings(**ratings_data),
            MISC=Misc(**misc_data),
        )

        assert performance.PLAYER_NAME == "LeBron James"
        assert performance.TEAM_ABBREVIATION == "LAL"
        assert performance.ABSOLUTE_STATISTICS.PTS == 25
