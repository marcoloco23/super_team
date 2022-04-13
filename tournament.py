import pymongo

client = pymongo.MongoClient(
    "mongodb+srv://superteam:4NgVPcNjmKBQkMTd@cluster0.sfhws.mongodb.net/dev?retryWrites=true&w=majority"
)
db = client.superteam
import pandas as pd
from nba_api.stats.static import players
from helpers import (
    flatten_performance_df,
    get_average_player_performances,
    stack_df,
)
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from nba_api.stats.endpoints import leaguegamefinder

print("Loading Model...")

model = xgb.XGBRegressor()
model.load_model("models/13_player_model.json")

gamefinder = leaguegamefinder.LeagueGameFinder()
all_games = gamefinder.get_data_frames()[0]
current_season = all_games[all_games.SEASON_ID == "22021"]
games = list(set(current_season.GAME_ID))
active_players = players.get_active_players()
active_players = pd.DataFrame(active_players)
active_player_ids = active_players.id.to_list()

print("Loading Performances...")

active_player_performances = pd.DataFrame(
    list(
        db.playerPerformances.find(
            {
                "PLAYER_ID": {"$in": active_player_ids},
                "GAME_DATE": {"$gte": "2022-01-01"},
            }
        )
    )
).set_index("_id")
performances = flatten_performance_df(active_player_performances)

season_performances = performances[performances.GAME_ID.isin(games)]
average_performances = get_average_player_performances(season_performances)


def run_tournament(performances, rounds=1, team_count=32, team_size=13):
    winner = False
    winner_list = []

    for _ in tqdm(range(rounds)):
        player_pool = performances[["PLAYER_ID", "PLAYER_NAME"]]
        team_list = []
        team_number = team_count

        if winner:
            player_pool.drop(winner_team.index)
            team_list.append(winner_team)
            team_number = team_number - 1

        for n in range(team_number):
            player_ids = player_pool.sample(team_size).PLAYER_ID
            team = performances[performances["PLAYER_ID"].isin(player_ids)]
            player_pool = player_pool.drop(team.index)
            team_list.append(team)

        for i in range(int(np.log2(team_count))):
            it = iter(team_list)
            team_list = []
            for (teamA, teamB) in zip(it, it):
                team_A_features = teamA.iloc[:, 2:].reset_index(drop=True)
                team_B_features = teamB.iloc[:, 2:].reset_index(drop=True)

                print(
                    "Team A: ",
                    teamA.sort_values("MIN", ascending=False).PLAYER_NAME.to_list(),
                    "\nTeam B: ",
                    teamB.sort_values("MIN", ascending=False).PLAYER_NAME.to_list(),
                )
                team_A_feature_df = pd.concat(
                    [
                        stack_df(
                            pd.concat([team_A_features, team_B_features]).reset_index(
                                drop=True
                            )
                        )
                    ],
                    axis=1,
                )
                team_B_feature_df = pd.concat(
                    [
                        stack_df(
                            pd.concat([team_B_features, team_A_features]).reset_index(
                                drop=True
                            )
                        )
                    ],
                    axis=1,
                )
                plus_minus_prediction = model.predict(
                    pd.concat([team_A_feature_df, team_B_feature_df])
                )

                if plus_minus_prediction[0] > plus_minus_prediction[1]:
                    team_list.append(teamA)
                    print("Team A wins")
                else:
                    team_list.append(teamB)
                    print("Team B wins")

        if len(team_list) == 1:
            winner_team = team_list[0]
            print(
                "Winner Team: ",
                winner_team.sort_values("MIN", ascending=False).PLAYER_NAME.to_list(),
            )
            winner = True
            winner_list.append(
                winner_team.sort_values("MIN", ascending=False).PLAYER_NAME.to_list()
            )

    return winner_list


winners = run_tournament(average_performances, rounds=10, team_size=13)
print(winners)
