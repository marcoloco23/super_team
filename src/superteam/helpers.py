"""
Helper functions for data processing and transformation.

This module provides utility functions for combining box scores,
processing player performance data, and calculating derived metrics.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from functools import reduce
from nba_api.stats.static import teams
from tqdm import tqdm
from nba_api.stats.endpoints import commonteamroster


def combine_team_games(
    df: pd.DataFrame,
    keep_method: str = "home"
) -> pd.DataFrame:
    """
    Combine team games by joining each row with opponents from the same game.

    Args:
        df: DataFrame containing game data with SEASON_ID, GAME_ID, GAME_DATE
        keep_method: Filter method - 'home', 'away', 'winner', 'loser', or None

    Returns:
        DataFrame with joined team data (columns suffixed with _A and _B)

    Raises:
        ValueError: If keep_method is not a valid option
    """
    # Join every row to all others with the same game ID
    joined = pd.merge(
        df, df, suffixes=["_A", "_B"], on=["SEASON_ID", "GAME_ID", "GAME_DATE"]
    )
    # Filter out any row that is joined to itself
    result = joined[joined.TEAM_ID_A != joined.TEAM_ID_B]

    # Take action based on the keep_method flag
    if keep_method is None:
        pass
    elif keep_method.lower() == "home":
        result = result[result.MATCHUP_A.str.contains(" vs. ")]
    elif keep_method.lower() == "away":
        result = result[result.MATCHUP_A.str.contains(" @ ")]
    elif keep_method.lower() == "winner":
        result = result[result.WL_A == "W"]
    elif keep_method.lower() == "loser":
        result = result[result.WL_A == "L"]
    else:
        raise ValueError(f"Invalid keep_method: {keep_method}")

    return result.reset_index(drop=True)


def get_combined_team_box_score(*box_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple team box score DataFrames into a single DataFrame.

    Args:
        *box_scores: Variable number of team box score DataFrames

    Returns:
        Combined DataFrame with all columns from input DataFrames
    """
    combined_box_score = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["TEAM_ID"],
            how="inner",
            suffixes=("", "_y"),
        ),
        box_scores,
    )
    combined_box_score.drop(
        combined_box_score.filter(regex="_y$").columns.tolist(), axis=1, inplace=True
    )
    combined_box_score = combined_box_score.loc[
        :, ~combined_box_score.columns.duplicated()
    ]
    combined_box_score.fillna(0, inplace=True)
    combined_box_score["TEAM_ABBREVIATION"] = combined_box_score[
        "TEAM_ABBREVIATION"
    ].astype("category")
    combined_box_score["MIN"] = combined_box_score["MIN"].apply(
        lambda time_str: float(time_str.replace(":", "."))
    )
    return combined_box_score


def get_combined_player_box_score(*box_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple player box score DataFrames into a single DataFrame.

    Args:
        *box_scores: Variable number of player box score DataFrames

    Returns:
        Combined DataFrame with all columns from input DataFrames
    """
    combined_box_score = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["PLAYER_ID", "TEAM_ID"],
            how="inner",
            suffixes=("", "_y"),
        ),
        box_scores,
    )
    combined_box_score.drop(
        combined_box_score.filter(regex="_y$").columns.tolist(), axis=1, inplace=True
    )
    combined_box_score = combined_box_score.loc[
        :, ~combined_box_score.columns.duplicated()
    ]
    combined_box_score.fillna(0, inplace=True)
    combined_box_score["TEAM_ABBREVIATION"] = combined_box_score[
        "TEAM_ABBREVIATION"
    ].astype("category")
    combined_box_score["MIN"] = combined_box_score["MIN"].apply(
        lambda time_str: float(time_str.replace(":", "."))
    )
    return combined_box_score


def get_player_and_team_box_scores(
    box_scores
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract player and team box scores from an API response.

    Args:
        box_scores: API response object with get_data_frames() method

    Returns:
        Tuple of (player_box_score, team_box_score) DataFrames
    """
    data_frames = box_scores.get_data_frames()
    return data_frames[0], data_frames[1]


def flatten_performance_df(performance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested performance data into a single-level DataFrame.

    Expands PERCENTAGES, ABSOLUTE_STATISTICS, RATINGS, and MISC columns.

    Args:
        performance_df: DataFrame with nested dictionary columns

    Returns:
        Flattened DataFrame with all nested columns expanded
    """
    i = performance_df.columns.get_loc("PERCENTAGES")
    percentages = performance_df.PERCENTAGES.apply(pd.Series)
    absolutes_stats = performance_df.ABSOLUTE_STATISTICS.apply(pd.Series)
    ratings = performance_df.RATINGS.apply(pd.Series)
    misc = performance_df.MISC.apply(pd.Series)

    performance_df = pd.concat(
        [performance_df.iloc[:, :i], percentages, absolutes_stats, ratings, misc],
        axis=1,
    )
    return performance_df


def get_performances_by_team(
    performance_df: pd.DataFrame,
    perf_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split performance data by team and prepare for model input.

    Args:
        performance_df: DataFrame containing game performance data
        perf_type: Type of performance - 'team' or 'player'

    Returns:
        Tuple of (team_1_performances, team_2_performances) DataFrames

    Raises:
        ValueError: If perf_type is not 'team' or 'player'
    """
    if perf_type == "team":
        start_col = 6
    elif perf_type == "player":
        start_col = 9
    else:
        raise ValueError(f"perf_type must be 'team' or 'player', got: {perf_type}")

    team_list = performance_df["TEAM_ABBREVIATION"].astype("category").cat.categories

    team_1_performances = performance_df[
        performance_df["TEAM_ABBREVIATION"] == team_list[0]
    ]
    team_2_performances = performance_df[
        performance_df["TEAM_ABBREVIATION"] == team_list[1]
    ]

    team_1_performances = (
        team_1_performances.iloc[:, start_col:]
        .apply(pd.to_numeric)
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )
    team_2_performances = (
        team_2_performances.iloc[:, start_col:]
        .apply(pd.to_numeric)
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )

    return team_1_performances, team_2_performances


def stack_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stack a DataFrame to create feature columns for model input.

    Transforms rows into columns with format {column}_{row_number}.

    Args:
        df: DataFrame to stack

    Returns:
        Single-row DataFrame with stacked features
    """
    stacked = df.copy()
    stacked.index = stacked.index + 1
    stacked = stacked.stack()
    stacked.index = stacked.index.map("{0[1]}_{0[0]}".format)
    stacked = stacked.to_frame().T
    return stacked


def win_loss_error_rate(
    test_predictions: np.ndarray,
    test_labels: pd.Series
) -> float:
    """
    Calculate the win/loss error rate for predictions.

    Args:
        test_predictions: Model predictions (positive = win)
        test_labels: Actual labels (positive = win)

    Returns:
        Error rate as a float between 0 and 1
    """
    win_loss_predictions = np.where(test_predictions > 0, 1, 0)
    win_loss_truth = np.where(test_labels.to_numpy() > 0, 1, 0)
    return abs(win_loss_predictions - win_loss_truth).mean()


def make_data_relative(x: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute statistics to relative differences between teams.

    Args:
        x: DataFrame with two rows (one per team)

    Returns:
        DataFrame with relative differences
    """
    diff_1 = x.iloc[0] - x.iloc[1]
    diff_2 = x.iloc[1] - x.iloc[0]
    x.iloc[0] = diff_1
    x.iloc[1] = diff_2
    return x


def get_average_player_performances(performances: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average performance statistics for each player.

    Args:
        performances: DataFrame with individual game performances

    Returns:
        DataFrame with averaged statistics per player
    """
    # Only average numeric columns
    numeric_cols = performances.select_dtypes(include=['number']).columns.tolist()
    # Keep PLAYER_ID and PLAYER_NAME for grouping
    cols_to_use = ["PLAYER_ID", "PLAYER_NAME"] + [c for c in numeric_cols if c not in ["PLAYER_ID"]]

    average_performances = performances[cols_to_use].groupby(
        ["PLAYER_ID", "PLAYER_NAME"]
    ).mean(numeric_only=True)
    # Fill NaN with 0 (some advanced stats may be missing for players with limited games)
    average_performances = average_performances.fillna(0).reset_index()

    # Drop TEAM_ID if it exists
    if "TEAM_ID" in average_performances.columns:
        average_performances = average_performances.drop("TEAM_ID", axis=1)

    # Get most recent team for each player (if TEAM_ABBREVIATION exists in original data)
    if "TEAM_ABBREVIATION" in performances.columns and "GAME_DATE" in performances.columns:
        # Sort by date and get the last team for each player
        sorted_perf = performances.sort_values("GAME_DATE")
        latest_teams = sorted_perf.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last()
        average_performances["TEAM_ABBREVIATION"] = average_performances["PLAYER_ID"].map(latest_teams)

        # Move TEAM_ABBREVIATION to the front (after PLAYER_ID, PLAYER_NAME)
        cols = average_performances.columns.tolist()
        cols.remove("TEAM_ABBREVIATION")
        # Insert after PLAYER_NAME (index 1) -> position 2
        cols.insert(2, "TEAM_ABBREVIATION")
        average_performances = average_performances[cols]

    return average_performances


def get_score_df(average_performances: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a composite player value score based on correlations with plus-minus.

    Args:
        average_performances: DataFrame with averaged player statistics

    Returns:
        DataFrame with player info and SCORE column, sorted by score
    """
    start_col = average_performances.columns.get_loc("PCT_FGA_2PT")
    score_df = average_performances.iloc[:, :start_col].copy()
    stats = average_performances.iloc[:, start_col:]

    # Normalize stats
    stats = stats - stats.min()
    stats = stats / stats.std()

    # Weight by correlation with plus-minus
    score_df["SCORE"] = stats.mul(stats.corrwith(stats.PLUS_MINUS)).mean(axis=1) ** 2
    score_df["SCORE"] = score_df["SCORE"] / score_df["SCORE"].max()
    score_df = score_df.sort_values("SCORE", axis=0, ascending=False).reset_index(
        drop=True
    )
    return score_df


def get_team_feature_df(
    team_A_features: pd.DataFrame,
    team_B_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Create feature DataFrame for team matchup prediction.

    Args:
        team_A_features: Feature DataFrame for team A
        team_B_features: Feature DataFrame for team B

    Returns:
        Combined and stacked feature DataFrame
    """
    team_feature_df = pd.concat(
        [
            stack_df(
                pd.concat([team_A_features, team_B_features]).reset_index(drop=True)
            )
        ],
        axis=1,
    )
    return team_feature_df


def get_salary_cap(
    average_performances: pd.DataFrame,
    team_size: int
) -> float:
    """
    Calculate the average team score to use as a salary cap.

    Args:
        average_performances: DataFrame with averaged player statistics
        team_size: Number of players per team

    Returns:
        Average total score per team (used as salary cap threshold)
    """
    score_df = get_score_df(average_performances)
    salary_cap_df = (
        score_df.groupby("TEAM_ABBREVIATION")
        .apply(lambda x: x[:team_size])
        .reset_index(drop=True)
    )
    salary_cap = salary_cap_df.groupby("TEAM_ABBREVIATION").sum(numeric_only=True).mean().SCORE
    return salary_cap


def insert_team_abbreviation(average_performances: pd.DataFrame) -> pd.DataFrame:
    """
    Add team abbreviation column to player performance data.

    Args:
        average_performances: DataFrame with player performance data

    Returns:
        DataFrame with TEAM_ABBREVIATION column added
    """
    player_team_dict = get_player_team_dict()
    average_performances["TEAM_ABBREVIATION"] = average_performances.PLAYER_ID.map(
        player_team_dict
    )
    average_performances = average_performances.dropna()
    first_column = average_performances.pop("TEAM_ABBREVIATION")
    average_performances.insert(0, "TEAM_ABBREVIATION", first_column)
    return average_performances


def get_player_team_dict() -> Dict[int, str]:
    """
    Build a mapping of player IDs to team abbreviations.

    Returns:
        Dictionary mapping player_id -> team_abbreviation
    """
    team_abbreviations = pd.DataFrame(teams.get_teams()).abbreviation.to_list()
    player_team_dict = {}
    for team_abb in tqdm(team_abbreviations, desc="Loading team rosters"):
        player_ids = get_team_player_ids(team_abb)
        for player_id in player_ids:
            player_team_dict[player_id] = team_abb
    return player_team_dict


def get_team_player_ids(team_abbreviation: str) -> List[int]:
    """
    Get list of player IDs for a team.

    Args:
        team_abbreviation: Team abbreviation (e.g., 'LAL', 'BOS')

    Returns:
        List of player IDs on the team roster
    """
    team_id = teams.find_team_by_abbreviation(team_abbreviation).get("id")
    team_players_df = commonteamroster.CommonTeamRoster(
        team_id=team_id
    ).get_data_frames()[0]
    return team_players_df.PLAYER_ID.to_list()
