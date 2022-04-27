import numpy as np
import pandas as pd
from functools import reduce
from nba_api.stats.static import teams
from tqdm import tqdm
from nba_api.stats.endpoints import commonteamroster


def combine_team_games(df, keep_method="home"):
    """Combine a TEAM_ID-GAME_ID unique table into rows by game. Slow.

        Parameters
        ----------
        df : Input DataFrame.
        keep_method : {'home', 'away', 'winner', 'loser', ``None``}, default 'home'
            - 'home' : Keep rows where TEAM_A is the home team.
            - 'away' : Keep rows where TEAM_A is the away team.
            - 'winner' : Keep rows where TEAM_A is the losing team.
            - 'loser' : Keep rows where TEAM_A is the winning team.
            - ``None`` : Keep all rows. Will result in an output DataFrame the same
                length as the input DataFrame.
                
        Returns
        -------
        result : DataFrame
    """
    # Join every row to all others with the same game ID.
    joined = pd.merge(
        df, df, suffixes=["_A", "_B"], on=["SEASON_ID", "GAME_ID", "GAME_DATE"]
    )
    # Filter out any row that is joined to itself.
    result = joined[joined.TEAM_ID_A != joined.TEAM_ID_B]
    # Take action based on the keep_method flag.
    if keep_method is None:
        # Return all the rows.
        pass
    elif keep_method.lower() == "home":
        # Keep rows where TEAM_A is the home team.
        result = result[result.MATCHUP_A.str.contains(" vs. ")]
    elif keep_method.lower() == "away":
        # Keep rows where TEAM_A is the away team.
        result = result[result.MATCHUP_A.str.contains(" @ ")]
    elif keep_method.lower() == "winner":
        result = result[result.WL_A == "W"]
    elif keep_method.lower() == "loser":
        result = result[result.WL_A == "L"]
    else:
        raise ValueError(f"Invalid keep_method: {keep_method}")
    return result.reset_index(drop=True)


def get_combined_team_box_score(*box_scores):
    combined_box_score = reduce(
        lambda left, right: pd.merge(
            left, right, on=["TEAM_ID"], how="inner", suffixes=("", "_y"),
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
        lambda time: float(time.replace(":", "."))
    )
    return combined_box_score


def get_combined_player_box_score(*box_scores):
    combined_box_score = reduce(
        lambda left, right: pd.merge(
            left, right, on=["PLAYER_ID", "TEAM_ID"], how="inner", suffixes=("", "_y"),
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
        lambda time: float(time.replace(":", "."))
    )
    return combined_box_score


def get_player_and_team_box_scores(box_scores):
    box_scores = box_scores.get_data_frames()
    box_score = box_scores[0]
    team_box_score = box_scores[1]
    return box_score, team_box_score


def flatten_performance_df(performance_df):
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


def get_performances_by_team(performance_df):
    if len(performance_df.columns) == 93:
        i = 6
    if len(performance_df.columns) == 114:
        i = 9
    team_list = performance_df["TEAM_ABBREVIATION"].astype("category").cat.categories
    team_1_performances = performance_df[
        performance_df["TEAM_ABBREVIATION"] == team_list[0]
    ]
    team_2_performances = performance_df[
        performance_df["TEAM_ABBREVIATION"] == team_list[1]
    ]
    team_1_performances = (
        team_1_performances.iloc[:, i:]
        .apply(pd.to_numeric)
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )
    team_2_performances = (
        team_2_performances.iloc[:, i:]
        .apply(pd.to_numeric)
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )
    return team_1_performances, team_2_performances


def stack_df(df):
    stack_df = df.copy()
    stack_df.index = stack_df.index + 1
    stack_df = stack_df.stack()
    stack_df.index = stack_df.index.map("{0[1]}_{0[0]}".format)
    stack_df = stack_df.to_frame().T
    return stack_df


def win_loss_error_rate(test_predictions, test_labels):
    win_loss_predictions = np.where(test_predictions > 0, 1, 0)
    win_loss_truth = np.where(test_labels.to_numpy() > 0, 1, 0)
    return abs(win_loss_predictions - win_loss_truth).mean()


def make_data_relative(x):
    diff_1 = x.iloc[0] - x.iloc[1]
    diff_2 = x.iloc[1] - x.iloc[0]
    x.iloc[0] = diff_1
    x.iloc[1] = diff_2
    return x


def get_average_player_performances(performances):
    average_performances = performances.groupby(
        ["PLAYER_ID", "PLAYER_NAME"], axis=0
    ).mean()
    average_performances = (
        average_performances.dropna().reset_index().drop("TEAM_ID", axis=1)
    )
    return average_performances


def get_score_df(average_performances):
    start_col = average_performances.columns.get_loc("PCT_FGA_2PT")
    score_df = average_performances.iloc[:, :start_col].copy()
    stats = average_performances.iloc[:, start_col:]
    stats = stats - stats.min()
    stats = stats / stats.std()
    score_df["SCORE"] = stats.mul(stats.corrwith(stats.PLUS_MINUS)).mean(axis=1) ** 2
    score_df["SCORE"] = score_df["SCORE"] / score_df["SCORE"].max()
    score_df = score_df.sort_values("SCORE", axis=0, ascending=False).reset_index(
        drop=True
    )
    return score_df


def get_team_feature_df(team_A_features, team_B_features):
    team_feature_df = pd.concat(
        [
            stack_df(
                pd.concat([team_A_features, team_B_features]).reset_index(drop=True)
            )
        ],
        axis=1,
    )
    return team_feature_df


def get_salary_cap(average_performances, team_size):
    score_df = get_score_df(average_performances)
    salary_cap_df = (
        score_df.groupby("TEAM_ABBREVIATION")
        .apply(lambda x: x[:team_size])
        .reset_index(drop=True)
    )
    salary_cap = salary_cap_df.groupby("TEAM_ABBREVIATION").sum().mean().SCORE
    return salary_cap


def insert_team_abbreviation(average_performances):
    player_team_dict = get_player_team_dict()
    average_performances["TEAM_ABBREVIATION"] = average_performances.PLAYER_ID.map(
        player_team_dict
    )
    average_performances = average_performances.dropna()
    first_column = average_performances.pop("TEAM_ABBREVIATION")
    average_performances.insert(0, "TEAM_ABBREVIATION", first_column)
    return average_performances


def get_player_team_dict():
    team_abbreviations = pd.DataFrame(teams.get_teams()).abbreviation.to_list()
    player_team_dict = {}
    for team_abb in tqdm(team_abbreviations):
        player_ids = get_team_player_ids(team_abb)
        for player_id in player_ids:
            player_team_dict[player_id] = team_abb
    return player_team_dict


def get_team_player_ids(team_abbreviation):
    team_id = teams.find_team_by_abbreviation(team_abbreviation).get("id")
    team_players_df = commonteamroster.CommonTeamRoster(
        team_id=team_id
    ).get_data_frames()[0]
    team_player_ids = team_players_df.PLAYER_ID.to_list()
    return team_player_ids
