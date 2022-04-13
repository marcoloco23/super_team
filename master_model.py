import pymongo
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import xgboost as xgb
from helpers import flatten_performance_df, make_data_relative, win_loss_error_rate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

client = pymongo.MongoClient(
    "mongodb+srv://superteam:4NgVPcNjmKBQkMTd@cluster0.sfhws.mongodb.net/dev?retryWrites=true&w=majority"
)
db = client.superteam

team_performances = pd.DataFrame(list(db.teamPerformances.find({}))).set_index("_id")

team_performances = flatten_performance_df(team_performances)
dataset = team_performances.loc[~(team_performances == 0).all(axis=1)]

target = "PLUS_MINUS"

features = dataset.iloc[:, 6:].copy()
labels = features.pop(target)

relative_features = features.copy()
relative_features["GAME_ID"] = dataset.GAME_ID
relative_features = relative_features.groupby("GAME_ID").apply(
    lambda x: make_data_relative(x)
)
relative_features = relative_features.drop(['MIN','PACE_PER40','PACE','E_PACE','PTS'],axis=1)
train_features, test_features, train_labels, test_labels = train_test_split(
    relative_features.drop("PTS", axis=1), labels, test_size=0.2, random_state=1
)

train_features, validation_features, train_labels, validation_labels = train_test_split(
    train_features, train_labels, test_size=0.25, random_state=1
)

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
model = model.fit(train_features, train_labels, eval_set=eval_set, verbose=True,early_stopping_rounds=50)

model.save_model("models/master_model.json")

test_predictions = model.predict(test_features)

wler = win_loss_error_rate(test_predictions, test_labels)
r2 = r2_score(test_labels, test_predictions)
print("Win Loss Accuracy: %f\n" % (1 - wler))
print("R^2: %f\n" % (r2))
