"""
Simulation and optimization module for NBA team matchups.

This module provides functions for simulating games between NBA teams,
running tournaments, finding optimal team compositions, and suggesting trades.
"""

from typing import List, Optional, Tuple, Dict
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from .helpers import get_salary_cap, get_score_df, get_team_feature_df
from nba_api.stats.static import players, teams


def _filter_features_for_model(feature_df: pd.DataFrame, model) -> pd.DataFrame:
    """Filter feature DataFrame to match model's expected columns."""
    expected_features = model.get_booster().feature_names
    return feature_df.reindex(columns=expected_features, fill_value=0)


def simulate_nba_matchup(
    team_abbreviation_A: str,
    team_abbreviation_B: str,
    average_performances: pd.DataFrame,
    model,
    team_A_injured_player_ids: Optional[List[int]] = None,
    team_B_injured_player_ids: Optional[List[int]] = None,
    team_size: int = 13,
) -> np.ndarray:
    """
    Simulate a matchup between two NBA teams.

    Args:
        team_abbreviation_A: Abbreviation for team A (e.g., 'LAL')
        team_abbreviation_B: Abbreviation for team B (e.g., 'BOS')
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        team_A_injured_player_ids: List of injured player IDs for team A
        team_B_injured_player_ids: List of injured player IDs for team B
        team_size: Number of players per team

    Returns:
        Array of [team_A_plus_minus, team_B_plus_minus] predictions
    """
    start_col = average_performances.columns.get_loc("PCT_FGA_2PT")
    team_A_player_ids = average_performances[
        average_performances.TEAM_ABBREVIATION == team_abbreviation_A
    ].PLAYER_ID.to_list()
    if team_A_injured_player_ids:
        team_A_player_ids = [
            player_id
            for player_id in team_A_player_ids
            if player_id not in team_A_injured_player_ids
        ]
    team_A_features = average_performances[
        average_performances.PLAYER_ID.isin(team_A_player_ids)
    ]
    team_A_features = (
        team_A_features.sort_values("MIN", ascending=False)
        .iloc[:team_size, start_col:]
        .reset_index(drop=True)
    )
    team_B_player_ids = average_performances[
        average_performances.TEAM_ABBREVIATION == team_abbreviation_B
    ].PLAYER_ID.to_list()
    if team_B_injured_player_ids:
        team_B_player_ids = [
            player_id
            for player_id in team_B_player_ids
            if player_id not in team_B_injured_player_ids
        ]
    team_B_features = average_performances[
        average_performances.PLAYER_ID.isin(team_B_player_ids)
    ]
    team_B_features = (
        team_B_features.sort_values("MIN", ascending=False)
        .iloc[:team_size, start_col:]
        .reset_index(drop=True)
    )

    team_A_feature_df = get_team_feature_df(team_A_features, team_B_features)
    team_B_feature_df = get_team_feature_df(team_B_features, team_A_features)

    # Filter to model's expected columns
    team_A_feature_df = _filter_features_for_model(team_A_feature_df, model)
    team_B_feature_df = _filter_features_for_model(team_B_feature_df, model)

    plus_minus_prediction = model.predict(
        pd.concat([team_A_feature_df, team_B_feature_df])
    )
    return plus_minus_prediction


def simulate_arbitrary_matchup(
    team_a_player_ids: List[int],
    team_b_player_ids: List[int],
    average_performances: pd.DataFrame,
    model,
    team_size: int = 13,
) -> np.ndarray:
    """
    Simulate a matchup between two custom teams.

    Args:
        team_a_player_ids: List of player IDs for team A
        team_b_player_ids: List of player IDs for team B
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        team_size: Number of players per team

    Returns:
        Array of [team_A_plus_minus, team_B_plus_minus] predictions
    """
    start_col = average_performances.columns.get_loc("PCT_FGA_2PT")
    team_A_features = average_performances[
        average_performances.PLAYER_ID.isin(team_a_player_ids)
    ]
    team_A_features = (
        team_A_features.sort_values("MIN", ascending=False)
        .iloc[:team_size, start_col:]
        .reset_index(drop=True)
    )

    team_B_features = average_performances[
        average_performances.PLAYER_ID.isin(team_b_player_ids)
    ]
    team_B_features = (
        team_B_features.sort_values("MIN", ascending=False)
        .iloc[:team_size, start_col:]
        .reset_index(drop=True)
    )

    team_A_feature_df = get_team_feature_df(team_A_features, team_B_features)
    team_B_feature_df = get_team_feature_df(team_B_features, team_A_features)

    # Filter to model's expected columns
    team_A_feature_df = _filter_features_for_model(team_A_feature_df, model)
    team_B_feature_df = _filter_features_for_model(team_B_feature_df, model)

    plus_minus_prediction = model.predict(
        pd.concat([team_A_feature_df, team_B_feature_df])
    )
    return plus_minus_prediction


def simulate_regular_season(average_performances, model, teams_df=None, team_size=13):
    """
    Simulate a regular season between all NBA teams.

    Args:
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        teams_df: Optional DataFrame of teams (if None, fetches from NBA API)
        team_size: Number of players per team

    Returns:
        Dictionary mapping team abbreviations to win ratios, sorted descending
    """
    if teams_df is not None:
        team_abbreviations = teams_df.abbreviation.to_list()
    else:
        team_abbreviations = pd.DataFrame(teams.get_teams()).abbreviation.to_list()

    results_dict = {}
    for i, team_A in tqdm(enumerate(team_abbreviations), total=len(team_abbreviations)):
        win_loss_list = []
        for team_B in [*team_abbreviations[:i], *team_abbreviations[i + 1 :]]:
            plus_minus_prediction = simulate_nba_matchup(
                team_A, team_B, average_performances, model=model, team_size=team_size
            )
            if plus_minus_prediction[0] > plus_minus_prediction[1]:
                win_loss_list.append(1)
            else:
                win_loss_list.append(0)

        results_dict[team_A] = np.mean(win_loss_list)
    return dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))


def run_tournament(average_performances, model, rounds=1, team_count=16, team_size=13):
    """
    Run a tournament bracket simulation with random teams.

    Args:
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        rounds: Number of tournament rounds to run
        team_count: Number of teams in the tournament (must be a power of 2)
        team_size: Number of players per team

    Returns:
        Tuple of (winner_name_list, winner_id_list) containing player names and IDs
    """
    winner = False
    winner_name_list = []
    winner_id_list = []

    for _ in tqdm(range(rounds)):
        player_pool = average_performances[["PLAYER_ID", "PLAYER_NAME"]]
        team_list = []
        team_number = team_count

        if winner:
            player_pool = player_pool.drop(winner_team.index)
            team_list.append(winner_team)
            team_number = team_number - 1

        for _ in range(team_number):
            player_ids = player_pool.sample(team_size, replace=False).PLAYER_ID
            team = average_performances[
                average_performances["PLAYER_ID"].isin(player_ids)
            ].drop_duplicates("PLAYER_NAME")
            team_list.append(team)

        for _ in range(int(np.log2(team_count))):
            it = iter(team_list)
            team_list = []
            for teamA, teamB in zip(it, it):
                team_A_ids = teamA.PLAYER_ID.to_list()
                team_B_ids = teamB.PLAYER_ID.to_list()
                plus_minus_prediction = simulate_arbitrary_matchup(
                    team_A_ids,
                    team_B_ids,
                    average_performances=average_performances,
                    model=model,
                    team_size=team_size,
                )

                if plus_minus_prediction[0] > plus_minus_prediction[1]:
                    team_list.append(teamA)
                else:
                    team_list.append(teamB)

        if len(team_list) == 1:
            winner_team = team_list[0]
            winner = True
            winner_name_list.append(
                winner_team.sort_values("MIN", ascending=False).PLAYER_NAME.to_list()
            )
            winner_id_list.append(
                winner_team.sort_values("MIN", ascending=False).PLAYER_ID.to_list()
            )

    return winner_name_list, winner_id_list


def evaluate_team(
    team_player_ids: List[int],
    average_performances: pd.DataFrame,
    model,
    team_size: int = 13,
    iterations: int = 100,
) -> float:
    """
    Evaluate a team's strength by simulating games against random opponents.

    Args:
        team_player_ids: List of player IDs for the team to test
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        team_size: Number of players per team
        iterations: Number of games to simulate

    Returns:
        Win rate as a float between 0 and 1
    """
    win_loss_list = []
    better_teams = []
    for _ in tqdm(range(iterations)):
        player_pool = average_performances[
            ~average_performances.PLAYER_ID.isin(team_player_ids)
        ].drop_duplicates()
        team_B_player_ids = player_pool.sample(
            team_size, replace=False
        ).PLAYER_ID.to_list()
        assert len(team_player_ids) == team_size
        assert len(team_B_player_ids) == team_size
        plus_minus_prediction = simulate_arbitrary_matchup(
            team_player_ids,
            team_B_player_ids,
            average_performances,
            model=model,
            team_size=team_size,
        )
        if plus_minus_prediction[0] > plus_minus_prediction[1]:
            win_loss_list.append(1)
        else:
            win_loss_list.append(0)
            better_teams.append(team_B_player_ids)
    return np.mean(win_loss_list)


def get_super_team(
    average_performances, model, team_size=13, iterations=100, salary_cap=True
):
    """
    Find the best team composition through iterative random sampling.

    Uses a genetic algorithm-style approach where random teams that beat the
    current best team (and meet salary cap constraints) become the new best.

    Args:
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        team_size: Number of players per team
        iterations: Number of random teams to test
        salary_cap: Whether to enforce salary cap constraints

    Returns:
        List of player IDs for the best team found
    """
    best_team = average_performances.sample(team_size).PLAYER_ID.to_list()
    score_df = get_score_df(average_performances)
    value_score = get_salary_cap(average_performances, 8)
    if not salary_cap:
        value_score = 1000

    for _ in tqdm(range(iterations)):
        challenger_team = average_performances.sample(team_size).PLAYER_ID.to_list()
        plus_minus_prediction = simulate_arbitrary_matchup(
            best_team,
            challenger_team,
            average_performances=average_performances,
            model=model,
            team_size=team_size,
        )
        team_value_score = (
            score_df[score_df.PLAYER_ID.isin(challenger_team)].fillna(0.5).sum().SCORE
        )
        # If challenger beats current best and is within salary cap, update best team
        if plus_minus_prediction[1] > plus_minus_prediction[0]:
            if team_value_score < value_score:
                best_team = challenger_team

    return best_team


def nba_test_team(
    team_player_ids: List[int],
    average_performances: pd.DataFrame,
    model,
    team_size: int = 13,
) -> float:
    """
    Test a team against all current NBA teams.

    Args:
        team_player_ids: List of player IDs for the team to test
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        team_size: Number of players per team

    Returns:
        Win rate against NBA teams as a float between 0 and 1
    """
    win_loss_list = []
    better_teams = []
    team_list = pd.DataFrame(teams.get_teams()).abbreviation.to_list()
    for team in team_list:
        team_B_player_ids = average_performances[
            average_performances.TEAM_ABBREVIATION == team
        ].PLAYER_ID.to_list()
        plus_minus_prediction = simulate_arbitrary_matchup(
            team_player_ids,
            team_B_player_ids,
            average_performances,
            model=model,
            team_size=team_size,
        )
        if plus_minus_prediction[0] > plus_minus_prediction[1]:
            win_loss_list.append(1)
        else:
            win_loss_list.append(0)
            better_teams.append(team_B_player_ids)
    return np.mean(win_loss_list)


def trade_finder(
    team_abbreviation: str,
    trade_value_df: pd.DataFrame,
    average_performances: pd.DataFrame,
    samples: int = 10,
    iterations: int = 100,
    team_size: int = 13,
) -> None:
    """
    Find potential trades to improve a team (prints results).

    Note: This is a legacy function that prints results. Use nba_trade_finder
    for a function that returns values.

    Args:
        team_abbreviation: Team abbreviation (e.g., 'LAL')
        trade_value_df: DataFrame with player trade values
        average_performances: DataFrame with player performance data
        samples: Number of trade scenarios to test
        iterations: Number of games per scenario
        team_size: Number of players per team
    """
    score_list = []
    trade_list = []
    team = list(
        average_performances[
            average_performances.TEAM_ABBREVIATION == team_abbreviation
        ]
        .sort_values("MIN", ascending=False)
        .reset_index()
        .PLAYER_ID[:team_size]
    )
    base_score = test_team(team, iterations=iterations, team_size=team_size)
    for _ in tqdm(range(samples)):
        new_team = team[:]
        traded_player = random.choice(team)
        new_team.remove(traded_player)
        trade_value = trade_value_df[
            trade_value_df.PLAYER_ID == traded_player
        ].SCORE.values[0]
        player_pool = average_performances[
            ~average_performances["PLAYER_ID"].isin(team)
        ]
        similar_valued_players = list(
            trade_value_df[
                trade_value_df.SCORE.between(trade_value - 0.1, trade_value + 0.1)
            ].PLAYER_ID
        )
        player_pool = player_pool[player_pool.PLAYER_ID.isin(similar_valued_players)]
        new_player = player_pool.sample(1).PLAYER_ID.to_list()[0]
        new_team.append(new_player)

        score = test_team(new_team, iterations=iterations, team_size=team_size)
        if score > base_score:
            score_list.append(score)
            trade_list.append((traded_player, new_player))

    if score_list:
        best_trade = trade_list[np.argmax(score_list)]
        traded_player_name = players.find_player_by_id(best_trade[0]).get("full_name")
        acquired_player_name = players.find_player_by_id(best_trade[1]).get("full_name")

        print(
            f"Trade {traded_player_name} for {acquired_player_name} to improve from {round(base_score,2)} to {round(max(score_list),2)} W/L"
        )
    else:
        print("No improvements found")


def nba_trade_finder(
    team_abbreviation: str,
    average_performances: pd.DataFrame,
    model,
    trade_player_id: Optional[int] = None,
    trade_threshold: float = 0.05,
    samples: int = 10,
    team_size: int = 13,
) -> Tuple[str, str, float, float]:
    """
    Find trades to improve a team by testing against all NBA teams.

    Args:
        team_abbreviation: Team abbreviation (e.g., 'LAL')
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        trade_player_id: Specific player to trade (if None, random selection)
        trade_threshold: Value range for finding similar-valued players
        samples: Number of trade scenarios to test
        team_size: Number of players per team

    Returns:
        Tuple of (traded_player_name, acquired_player_name, base_score, best_score)
    """
    trade_value_df = get_score_df(average_performances)
    score_list = []
    trade_list = []
    team_player_ids = average_performances[
        average_performances.TEAM_ABBREVIATION == team_abbreviation
    ].PLAYER_ID.to_list()
    team_features = average_performances[
        average_performances.PLAYER_ID.isin(team_player_ids)
    ]
    team = (
        team_features.sort_values("MIN", ascending=False)
        .PLAYER_ID[:team_size]
        .to_list()
    )
    base_score = nba_test_team(
        team, average_performances, model=model, team_size=team_size
    )
    for _ in tqdm(range(samples)):
        new_team = team[:]
        if trade_player_id:
            traded_player = trade_player_id
        else:
            traded_player = random.choice(team)
        new_team.remove(traded_player)
        trade_value = trade_value_df[
            trade_value_df.PLAYER_ID == traded_player
        ].SCORE.values[0]
        player_pool = average_performances[
            ~average_performances["PLAYER_ID"].isin(team)
        ]
        similar_valued_players = list(
            trade_value_df[
                trade_value_df.SCORE.between(
                    trade_value - trade_threshold, trade_value + trade_threshold
                )
            ].PLAYER_ID
        )
        player_pool = player_pool[player_pool.PLAYER_ID.isin(similar_valued_players)]
        new_player = player_pool.sample(1).PLAYER_ID.to_list()[0]
        new_team.append(new_player)
        score = nba_test_team(
            new_team, average_performances, model=model, team_size=team_size
        )
        score_list.append(score)
        trade_list.append((traded_player, new_player))

    best_trade = trade_list[np.argmax(score_list)]
    traded_player_name = players.find_player_by_id(best_trade[0]).get("full_name")
    acquired_player_name = players.find_player_by_id(best_trade[1]).get("full_name")

    return traded_player_name, acquired_player_name, base_score, max(score_list)


def build_team_around_player(
    player_name,
    average_performances,
    model,
    team_size=13,
    iterations=10,
    salary_cap=True,
):
    """
    Build the best team around a specified player.

    Args:
        player_name: Name of the player to build the team around
        average_performances: DataFrame with player performance data
        model: Trained XGBoost model for predictions
        team_size: Number of players per team
        iterations: Number of random teams to test
        salary_cap: Whether to enforce salary cap constraints

    Returns:
        List of player IDs for the best team found (always includes the specified player)
    """
    player_id = average_performances[
        average_performances.PLAYER_NAME == player_name
    ].PLAYER_ID
    player_pool = average_performances[
        ~average_performances["PLAYER_ID"].isin(player_id)
    ]

    # Initialize with the anchor player plus random teammates
    best_team = (
        player_pool.sample(team_size - 1, replace=False)
        .PLAYER_ID.drop_duplicates()
        .to_list()
    )
    best_team = best_team + player_id.to_list()

    score_df = get_score_df(average_performances)
    value_score = get_salary_cap(average_performances, team_size)
    if not salary_cap:
        value_score = 1000

    for _ in tqdm(range(iterations)):
        # Create challenger team with same anchor player
        challenger_teammates = player_pool.sample(
            team_size - 1, replace=False
        ).PLAYER_ID.to_list()
        challenger_team = challenger_teammates + player_id.to_list()

        plus_minus_prediction = simulate_arbitrary_matchup(
            best_team,
            challenger_team,
            average_performances=average_performances,
            model=model,
            team_size=team_size,
        )
        team_value_score = (
            score_df[score_df.PLAYER_ID.isin(challenger_team)].fillna(0.5).sum().SCORE
        )
        # If challenger beats current best and is within salary cap, update best team
        if plus_minus_prediction[1] > plus_minus_prediction[0]:
            if team_value_score < value_score:
                best_team = challenger_team

    return best_team
