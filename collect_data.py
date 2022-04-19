import time
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoreplayertrackv2,
    boxscoreadvancedv2,
    boxscoretraditionalv2,
    boxscorefourfactorsv2,
    boxscoremiscv2,
    boxscorescoringv2,
    boxscoreusagev2,
)
from nba_api.stats.static import teams
import pandas as pd
from models import (
    AbsoluteStatistics,
    Misc,
    PlayerPercentages,
    PlayerPerformance,
    Ratings,
    TeamPercentages,
    TeamPerformance,
)
from tqdm import tqdm
import pymongo
from constants import MONGO_DB, MONGO_NAME, MONGO_PW
from helpers import (
    get_combined_player_box_score,
    get_combined_team_box_score,
    get_player_and_team_box_scores,
)

pd.options.mode.chained_assignment = None


client = pymongo.MongoClient(
    f"mongodb+srv://{MONGO_NAME}:{MONGO_PW}@cluster0.sfhws.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"
)
db = client.superteam

upload_count = 100


nba_teams = teams.get_teams()
team_df = pd.DataFrame(nba_teams)

print("Loading Games...")
nba_games = leaguegamefinder.LeagueGameFinder(
    league_id_nullable="00"
).get_data_frames()[0]
game_ids = set(nba_games.GAME_ID.to_list())

print("Loading Database...")
existing_player_performances = list(
    db.playerPerformances.find({}, projection=["GAME_ID", "PLAYER_ID"])
)
existing_team_performances = list(
    db.teamPerformances.find({}, projection=["GAME_ID", "TEAM_ID"])
)
new_player_performances, new_team_performances = [], []

counter = 0

if not pd.DataFrame(existing_team_performances).empty:
    existing_game_ids = list(set(pd.DataFrame(existing_team_performances).GAME_ID))
else:
    existing_game_ids = []

for i, game_id in tqdm(enumerate(list(game_ids)), total=len(list(game_ids))):
    game_date = list(set(nba_games[nba_games.GAME_ID == game_id].GAME_DATE))[0]
    if counter == upload_count:
        print("Uploading Data...")
        db.playerPerformances.insert_many(new_player_performances)
        db.teamPerformances.insert_many(new_team_performances)
        time.sleep(1)

        # Initialize
        existing_performances = list(
            db.playerPerformances.find({}, projection=["GAME_ID", "PLAYER_ID"])
        )
        existing_team_performances = list(
            db.teamPerformances.find({}, projection=["GAME_ID", "TEAM_ID"])
        )

        existing_game_ids = list(set(pd.DataFrame(existing_team_performances).GAME_ID))

        new_player_performances, new_team_performances = [], []
        counter = 0

    if game_id in existing_game_ids:
        continue

    # Api Calls
    try:
        advanced_box_scores = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id)
        advanced_box_score, advanced_team_box_score = get_player_and_team_box_scores(
            advanced_box_scores
        )

        basic_box_scores = boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id)
        basic_box_score, basic_team_box_score = get_player_and_team_box_scores(
            basic_box_scores
        )

        traditional_box_scores = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id)
        (
            traditional_box_score,
            traditional_team_box_score,
        ) = get_player_and_team_box_scores(traditional_box_scores)

        four_factors_box_scores = boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id)
        (
            four_factors_box_score,
            four_factors_team_box_score,
        ) = get_player_and_team_box_scores(four_factors_box_scores)

        misc_box_scores = boxscoremiscv2.BoxScoreMiscV2(game_id)
        misc_box_score, misc_team_box_score = get_player_and_team_box_scores(
            misc_box_scores
        )

        scoring_box_scores = boxscorescoringv2.BoxScoreScoringV2(game_id)
        scoring_box_score, scoring_team_box_score = get_player_and_team_box_scores(
            scoring_box_scores
        )

        usage_box_scores = boxscoreusagev2.BoxScoreUsageV2(game_id)
        usage_box_score, usage_team_box_score = get_player_and_team_box_scores(
            usage_box_scores
        )

    except Exception as e:
        print(e)
        continue

    if basic_box_score.empty or advanced_box_score.empty or traditional_box_score.empty:
        continue

    try:
        combined_box_score = get_combined_player_box_score(
            basic_box_score,
            advanced_box_score,
            traditional_box_score,
            four_factors_box_score,
            misc_box_score,
            scoring_box_score,
            usage_box_score,
        )
        combined_team_box_score = get_combined_team_box_score(
            basic_team_box_score,
            advanced_team_box_score,
            traditional_team_box_score,
            four_factors_team_box_score,
            misc_team_box_score,
            scoring_team_box_score,
            usage_team_box_score,
        )
    except Exception as e:
        print(e)
        continue

    if combined_box_score.empty or combined_team_box_score.empty:
        continue

    combined_box_score = combined_box_score.drop_duplicates()
    combined_team_box_score = combined_team_box_score.drop_duplicates()

    for i, row in combined_box_score.iterrows():
        performance = PlayerPerformance(
            **row,
            GAME_DATE=game_date,
            PERCENTAGES=PlayerPercentages(**row),
            ABSOLUTE_STATISTICS=AbsoluteStatistics(**row),
            RATINGS=Ratings(**row),
            MISC=Misc(**row),
        )
        existing_performance = next(
            (
                item
                for item in existing_player_performances
                if item["GAME_ID"] == performance.GAME_ID
                and item["PLAYER_ID"] == performance.PLAYER_ID
            ),
            None,
        )
        if not existing_performance:
            new_player_performances.append(performance.dict())

    for i, row in combined_team_box_score.iterrows():
        performance = TeamPerformance(
            **row,
            GAME_DATE=game_date,
            PERCENTAGES=TeamPercentages(**row),
            ABSOLUTE_STATISTICS=AbsoluteStatistics(**row),
            RATINGS=Ratings(**row),
            MISC=Misc(**row),
        )
        existing_performance = next(
            (
                item
                for item in existing_team_performances
                if item["GAME_ID"] == performance.GAME_ID
                and item["TEAM_ID"] == performance.TEAM_ID
            ),
            None,
        )
        if not existing_performance:
            new_team_performances.append(performance.dict())

    counter += 1

