from dataclasses import replace
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from helpers import get_team_feature_df
from nba_api.stats.static import players


def simulate_nba_matchup(
    team_abbreviation_A, team_abbreviation_B, average_performances, model, team_size=13
):
    i = average_performances.columns.get_loc("PCT_FGA_2PT")
    player_ids = list(
        average_performances[
            average_performances.TEAM_ABBREVIATION == team_abbreviation_A
        ]
        .groupby(["PLAYER_ID", "PLAYER_NAME"])
        .mean()
        .sort_values("MIN", ascending=False)
        .reset_index()
        .PLAYER_ID[:team_size]
    )
    team_A_features = average_performances[
        average_performances.PLAYER_ID.isin(player_ids)
    ]
    team_A_features = (
        team_A_features.iloc[:team_size, i:]
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )

    player_ids = list(
        average_performances[
            average_performances.TEAM_ABBREVIATION == team_abbreviation_B
        ]
        .groupby(["PLAYER_ID", "PLAYER_NAME"])
        .mean()
        .sort_values("MIN", ascending=False)
        .reset_index()
        .PLAYER_ID[:team_size]
    )
    team_B_features = average_performances[
        average_performances.PLAYER_ID.isin(player_ids)
    ]
    team_B_features = (
        team_B_features.iloc[:team_size, i:]
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )

    team_A_feature_df = get_team_feature_df(team_A_features, team_B_features)
    team_B_feature_df = get_team_feature_df(team_B_features, team_A_features)
    plus_minus_prediction = model.predict(
        pd.concat([team_A_feature_df, team_B_feature_df])
    )
    return plus_minus_prediction


def simulate_regular_season(average_performances, model, team_size=13):
    teams = average_performances.TEAM_ABBREVIATION.unique()
    results_dict = {}
    for i, team_A in tqdm(enumerate(teams), total=len(teams)):
        win_loss_list = []
        for team_B in [*teams[:i], *teams[i + 1 :]]:
            plus_minus_prediction = simulate_nba_matchup(
                team_A, team_B, average_performances, model=model, team_size=team_size
            )
            if plus_minus_prediction[0] > plus_minus_prediction[1]:
                win_loss_list.append(1)
            else:
                win_loss_list.append(0)

        results_dict[team_A] = np.mean(win_loss_list)

    return dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))


def simulate_arbitrary_matchup(
    team_a_player_ids, team_b_player_ids, average_performances, model, team_size=13
):
    start_col = average_performances.columns.get_loc("PCT_FGA_2PT")
    team_A_features = average_performances[
        average_performances.PLAYER_ID.isin(team_a_player_ids)
    ].drop_duplicates("PLAYER_NAME")
    team_A_features = (
        team_A_features.iloc[:team_size, start_col:]
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )

    team_B_features = average_performances[
        average_performances.PLAYER_ID.isin(team_b_player_ids)
    ].drop_duplicates("PLAYER_NAME")
    team_B_features = (
        team_B_features.iloc[:team_size, start_col:]
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )

    team_A_feature_df = get_team_feature_df(team_A_features, team_B_features)
    team_B_feature_df = get_team_feature_df(team_B_features, team_A_features)
    plus_minus_prediction = model.predict(
        pd.concat([team_A_feature_df, team_B_feature_df])
    )
    return plus_minus_prediction


def run_tournament(average_performances, model, rounds=1, team_count=16, team_size=13):
    winner = False
    winner_list = []

    for _ in tqdm(range(rounds)):
        player_pool = average_performances[["PLAYER_ID", "PLAYER_NAME"]]
        team_list = []
        team_number = team_count

        if winner:
            player_pool.drop(winner_team.index)
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
            for (teamA, teamB) in zip(it, it):
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
            print(
                "Winner Team: ",
                winner_team.sort_values("MIN", ascending=False).PLAYER_NAME.to_list(),
            )
            winner = True
            winner_list.append(
                winner_team.sort_values("MIN", ascending=False).PLAYER_ID.to_list()
            )

    return winner_list


def test_team(
    team_player_ids, average_performances, model, team_size=13, iterations=100
):
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
    average_performances, model, team_size=13, iterations=10, test_teams=100
):
    team_A_player_ids = average_performances.sample(team_size).PLAYER_ID.to_list()
    for i in range(iterations):
        if i > 0:
            if better_teams:
                team_A_player_ids = random.choice(better_teams)
            else:
                print("Super Team Found")
                break

        win_loss_list = []
        better_teams = []
        for _ in tqdm(range(test_teams)):
            team_B_player_ids = average_performances.sample(
                team_size
            ).PLAYER_ID.to_list()
            plus_minus_prediction = simulate_arbitrary_matchup(
                team_A_player_ids,
                team_B_player_ids,
                average_performances=average_performances,
                model=model,
                team_size=team_size,
            )
            if plus_minus_prediction[0] > plus_minus_prediction[1]:
                win_loss_list.append(1)
            else:
                win_loss_list.append(0)
                better_teams.append(team_B_player_ids)

        print("W/L: ", np.mean(win_loss_list))

    return team_A_player_ids


def nba_test_team(team_player_ids, average_performances, model, team_size=13):
    start_col = average_performances.columns.get_loc("PCT_FGA_2PT")
    teams = average_performances.TEAM_ABBREVIATION.unique()
    team_A_features = average_performances[
        average_performances["PLAYER_ID"].isin(team_player_ids)
    ].drop_duplicates("PLAYER_NAME")
    team_A_features = (
        team_A_features.iloc[:team_size, start_col:]
        .sort_values("MIN", ascending=False)
        .reset_index(drop=True)
    )
    win_loss_list = []
    better_teams = []
    for team in teams:
        team_B_player_ids = list(
            average_performances[average_performances.TEAM_ABBREVIATION == team]
            .sort_values("MIN", ascending=False)
            .reset_index()
            .PLAYER_ID[:team_size]
        )
        team_B_features = average_performances[
            average_performances["PLAYER_ID"].isin(team_B_player_ids)
        ]
        team_B_features = (
            team_B_features.iloc[:team_size, start_col:]
            .sort_values("MIN", ascending=False)
            .reset_index(drop=True)
        )
        team_A_feature_df = get_team_feature_df(team_A_features, team_B_features)
        team_B_feature_df = get_team_feature_df(team_B_features, team_A_features)
        plus_minus_prediction = model.predict(
            pd.concat([team_A_feature_df, team_B_feature_df])
        )
        if plus_minus_prediction[0] > plus_minus_prediction[1]:
            win_loss_list.append(1)
        else:
            win_loss_list.append(0)
            better_teams.append(team_B_player_ids)

    return np.mean(win_loss_list)


def nba_trade_finder(
    team_abbreviation,
    trade_value_df,
    average_performances,
    model,
    trade_player_id=None,
    samples=10,
    team_size=13,
):
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
                trade_value_df.SCORE.between(trade_value - 0.1, trade_value + 0.1)
            ].PLAYER_ID
        )
        player_pool = player_pool[player_pool.PLAYER_ID.isin(similar_valued_players)]
        new_player = player_pool.sample(1).PLAYER_ID.to_list()[0]
        new_team.append(new_player)

        score = nba_test_team(
            new_team, average_performances, model=model, team_size=team_size
        )
        if score > base_score:
            score_list.append(score)
            trade_list.append((traded_player, new_player))

    if score_list:
        best_trade = trade_list[np.argmax(score_list)]
        traded_player_name = players.find_player_by_id(best_trade[0]).get("full_name")
        acquired_player_name = players.find_player_by_id(best_trade[1]).get("full_name")

        print(
            f"Trade {traded_player_name} for {acquired_player_name} to improve from {round(base_score,3)} to {round(max(score_list),3)} W/L"
        )
    else:
        print("No improvements found")


def build_team_around_player(
    player_name,
    average_performances,
    model,
    iterations=10,
    test_teams=100,
    team_size=13,
):
    player = average_performances[
        average_performances.PLAYER_NAME == player_name
    ].PLAYER_ID
    player_pool = average_performances[~average_performances["PLAYER_ID"].isin(player)]
    max_score = 0
    for _ in tqdm(range(iterations)):
        team = (
            player_pool.sample(team_size - 1, replace=False)
            .PLAYER_ID.drop_duplicates()
            .to_list()
        )
        team = team + player.to_list()
        assert len(team) == team_size
        score = test_team(
            team,
            average_performances,
            model=model,
            team_size=team_size,
            iterations=test_teams,
        )
        if score > max_score:
            max_score = score
            best_team = team
    player_names = (
        average_performances[average_performances["PLAYER_ID"].isin(best_team)]
        .sort_values("MIN", ascending=False)
        .PLAYER_NAME.to_list()
    )
    return player_names, max_score


def get_super_team_for_player(average_performances, player_name, model, team_size=13):
    player = average_performances[
        average_performances.PLAYER_NAME == player_name
    ].PLAYER_ID
    player_pool = average_performances[~average_performances["PLAYER_ID"].isin(player)]
    team_A_player_ids = player_pool.sample(team_size - 1).PLAYER_ID.to_list()
    team_A_player_ids = team_A_player_ids + player.to_list()
    for i in range(10):
        if i > 0:
            if better_teams:
                team_A_player_ids = random.choice(better_teams)
            else:
                print("Super Team Found")
                break
        win_loss_list = []
        better_teams = []
        for _ in tqdm(range(100)):
            team_B_player_ids = player_pool.sample(team_size - 1).PLAYER_ID.to_list()
            team_B_player_ids = team_B_player_ids + player.to_list()
            plus_minus_prediction = simulate_arbitrary_matchup(
                team_A_player_ids,
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

        print("W/L: ", np.mean(win_loss_list))

    return team_A_player_ids