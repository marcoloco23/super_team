import numpy as np
import pandas as pd
from functools import reduce


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
