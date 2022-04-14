import pymongo

from constants import MONGO_NAME, MONGO_PW, MONGO_DB

client = pymongo.MongoClient(
    f"mongodb+srv://{MONGO_NAME}:{MONGO_PW}@cluster0.sfhws.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"
)
db = client.superteam
import pandas as pd
from helpers import (
    flatten_performance_df,
    get_performances_by_team,
    stack_df,
    win_loss_error_rate,
)
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

player_count = 13
target = "PLUS_MINUS"

print("Loading Data from Database...")
player_performances = db.playerPerformances.find({})
team_performances = db.teamPerformances.find({})
player_performance_df = pd.DataFrame(list(player_performances)).set_index("_id")
team_performance_df = pd.DataFrame(list(team_performances)).set_index("_id")
player_performance_df = flatten_performance_df(player_performance_df)
team_performance_df = flatten_performance_df(team_performance_df)


team_game_ids = list(set(team_performance_df.GAME_ID))
player_game_ids = list(set(player_performance_df.GAME_ID))
game_ids = list(set(team_game_ids) & set(player_game_ids))

print("Preproccessing Data...")
data_df_list = []
for game_id in tqdm(game_ids):
    game_player_performances = player_performance_df[
        player_performance_df.GAME_ID == game_id
    ].drop_duplicates()
    game_team_performances = team_performance_df[
        team_performance_df.GAME_ID == game_id
    ].drop_duplicates()
    a_player, b_player = get_performances_by_team(game_player_performances)
    a_team, b_team = get_performances_by_team(game_team_performances)

    team_a_feature_df = pd.concat(
        [
            stack_df(
                pd.concat(
                    [a_player[:player_count], b_player[:player_count]]
                ).reset_index(drop=True)
            )
        ],
        axis=1,
    )
    team_a_data_df = pd.concat([team_a_feature_df, a_team[target]], axis=1)

    team_b_feature_df = pd.concat(
        [
            stack_df(
                pd.concat(
                    [b_player[:player_count], a_player[:player_count]]
                ).reset_index(drop=True)
            )
        ],
        axis=1,
    )
    team_b_data_df = pd.concat([team_b_feature_df, b_team[target]], axis=1)

    data_df_list.append(team_a_data_df)
    data_df_list.append(team_b_data_df)


X = pd.concat(data_df_list).fillna(0).reset_index(drop=True)
y = X.pop(target)

train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=1
)
train_features, validation_features, train_labels, validation_labels = train_test_split(
    train_features, train_labels, test_size=0.25, random_state=1
)
print("Training Model...")
n = 5000
model = xgb.XGBRegressor(
    booster="gbtree",
    learning_rate=0.01,
    n_estimators=n,
    max_depth=4,
    min_child_weight=4,
    gamma=0.6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    nthread=-1,
    eval_metric="rmse",
)
eval_set = [(validation_features, validation_labels)]
model = model.fit(
    train_features,
    train_labels,
    eval_set=eval_set,
    verbose=True,
    early_stopping_rounds=50,
)
model.save_model(f"models/{player_count}_player_model.json")

predictions = model.predict(test_features)
wler = win_loss_error_rate(predictions, test_labels)
r2 = r2_score(test_labels, predictions)
print("Win Loss Accuracy: %f\n" % (1 - wler))
print("R^2: %f\n" % (r2))

