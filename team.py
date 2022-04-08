import pymongo
import warnings

from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from tqdm import tqdm
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor


from helpers import flatten_performance_df, get_performances_by_team, stack_df

starting_5 = False
print(f"Staring 5: {starting_5}")

client = pymongo.MongoClient(
    "mongodb+srv://superteam:4NgVPcNjmKBQkMTd@cluster0.sfhws.mongodb.net/dev?retryWrites=true&w=majority"
)
db = client.superteam

print("Loading Data...")
player_performances = pd.DataFrame(list(db.playerPerformances.find({}))).set_index(
    "_id"
)
team_performances = pd.DataFrame(list(db.teamPerformances.find({}))).set_index("_id")

player_performances = flatten_performance_df(player_performances)
team_performances = flatten_performance_df(team_performances)

game_ids = list(set(team_performances.GAME_ID))

feature_list, target_list = [], []
for game_id in tqdm(game_ids):
    player_game_performances = player_performances[
        player_performances.GAME_ID == game_id
    ]
    team_game_performances = team_performances[team_performances.GAME_ID == game_id]

    team_1_performances, team_2_performances = get_performances_by_team(
        team_game_performances
    )
    team_1_player_performances, team_2_player_performances = get_performances_by_team(
        player_game_performances
    )
    if starting_5:
        team_1_player_performances = team_1_player_performances[:5]
        team_2_player_performances = team_2_player_performances[:5]

    for i in [team_1_player_performances[:13], team_2_player_performances[:13]]:
        stacked_df = stack_df(i)
        feature_list.append(stacked_df)

    for i in [team_1_performances, team_2_performances]:
        target_list.append(i)

features = pd.concat(feature_list).fillna(0).reset_index(drop=True)
targets = pd.concat(target_list).fillna(0).reset_index(drop=True)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, targets, test_size=0.2, random_state=1
)

print("Training Models...")
model = xgb.XGBRegressor(
    booster="gbtree",
    learning_rate=0.01,
    n_estimators=100,
    max_depth=4,
    min_child_weight=4,
    gamma=0.6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    nthread=-1,
    eval_metric="rmse",
)
multioutputregressor = MultiOutputRegressor(model).fit(train_features, train_labels)
predictions = multioutputregressor.predict(test_features)
prediction_df = pd.DataFrame(predictions, columns=test_labels.columns)
print("Mean Squared Error:")
print(
    np.mean(
        (predictions - test_labels) ** 2, axis=0
    ).mean()
)
r2 = r2_score(test_labels, prediction_df)
print(r2)


joblib.dump(multioutputregressor, "models/team_model.pkl")
