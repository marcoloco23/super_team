"""
Data collection module for fetching NBA game statistics.

This module fetches box score data from the NBA API and stores it in MongoDB.
It handles multiple box score types and implements batch uploading for efficiency.
"""

import time
import logging

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
from pymongo.errors import ConnectionFailure, OperationFailure
import pymongo

from .models import (
    AbsoluteStatistics,
    Misc,
    PlayerPercentages,
    PlayerPerformance,
    Ratings,
    TeamPercentages,
    TeamPerformance,
)
from tqdm import tqdm
from .constants import MONGO_DB, MONGO_NAME, MONGO_PW
from .helpers import (
    get_combined_player_box_score,
    get_combined_team_box_score,
    get_player_and_team_box_scores,
)
from .logger import setup_logger

# Suppress pandas chained assignment warning
pd.options.mode.chained_assignment = None

# Set up logging
logger = setup_logger("collect_data")

# Configuration
UPLOAD_BATCH_SIZE = 100
API_RATE_LIMIT_DELAY = 0.6  # seconds between API calls


def get_mongo_client():
    """
    Create and return a MongoDB client connection.

    Returns:
        pymongo.MongoClient: Connected MongoDB client

    Raises:
        ConnectionFailure: If connection to MongoDB fails
    """
    try:
        client = pymongo.MongoClient(
            f"mongodb+srv://{MONGO_NAME}:{MONGO_PW}@cluster0.sfhws.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority",
            serverSelectionTimeoutMS=5000
        )
        # Test the connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


def fetch_box_scores(game_id: str) -> tuple:
    """
    Fetch all box score types for a given game.

    Args:
        game_id: NBA game ID

    Returns:
        Tuple of (player_box_scores, team_box_scores) where each is a tuple of DataFrames

    Raises:
        Exception: If any API call fails
    """
    # Add delay to respect rate limits
    time.sleep(API_RATE_LIMIT_DELAY)

    advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id)
    advanced_player, advanced_team = get_player_and_team_box_scores(advanced)

    basic = boxscoreplayertrackv2.BoxScorePlayerTrackV2(game_id)
    basic_player, basic_team = get_player_and_team_box_scores(basic)

    traditional = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id)
    traditional_player, traditional_team = get_player_and_team_box_scores(traditional)

    four_factors = boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id)
    four_factors_player, four_factors_team = get_player_and_team_box_scores(four_factors)

    misc = boxscoremiscv2.BoxScoreMiscV2(game_id)
    misc_player, misc_team = get_player_and_team_box_scores(misc)

    scoring = boxscorescoringv2.BoxScoreScoringV2(game_id)
    scoring_player, scoring_team = get_player_and_team_box_scores(scoring)

    usage = boxscoreusagev2.BoxScoreUsageV2(game_id)
    usage_player, usage_team = get_player_and_team_box_scores(usage)

    player_box_scores = (
        basic_player, advanced_player, traditional_player,
        four_factors_player, misc_player, scoring_player, usage_player
    )
    team_box_scores = (
        basic_team, advanced_team, traditional_team,
        four_factors_team, misc_team, scoring_team, usage_team
    )

    return player_box_scores, team_box_scores


def upload_batch(db, player_performances: list, team_performances: list) -> None:
    """
    Upload a batch of performances to MongoDB.

    Args:
        db: MongoDB database instance
        player_performances: List of player performance documents
        team_performances: List of team performance documents
    """
    try:
        if player_performances:
            db.playerPerformances.insert_many(player_performances)
            logger.info(f"Uploaded {len(player_performances)} player performances")

        if team_performances:
            db.teamPerformances.insert_many(team_performances)
            logger.info(f"Uploaded {len(team_performances)} team performances")

    except OperationFailure as e:
        logger.error(f"Failed to upload batch: {e}")
        raise


def collect_data():
    """
    Main data collection function.

    Fetches game data from NBA API and stores in MongoDB.
    Implements batch uploading and handles errors gracefully.
    """
    # Connect to MongoDB
    client = get_mongo_client()
    db = client.superteam

    # Get all NBA teams
    nba_teams = teams.get_teams()
    team_df = pd.DataFrame(nba_teams)
    logger.info(f"Loaded {len(nba_teams)} NBA teams")

    # Fetch all games
    logger.info("Loading games from NBA API...")
    nba_games = leaguegamefinder.LeagueGameFinder(
        league_id_nullable="00"
    ).get_data_frames()[0]
    game_ids = set(nba_games.GAME_ID.to_list())
    logger.info(f"Found {len(game_ids)} games")

    # Load existing data to avoid duplicates
    logger.info("Loading existing data from database...")
    existing_player_performances = list(
        db.playerPerformances.find({}, projection=["GAME_ID", "PLAYER_ID"])
    )
    existing_team_performances = list(
        db.teamPerformances.find({}, projection=["GAME_ID", "TEAM_ID"])
    )

    if not pd.DataFrame(existing_team_performances).empty:
        existing_game_ids = set(pd.DataFrame(existing_team_performances).GAME_ID)
    else:
        existing_game_ids = set()

    logger.info(f"Found {len(existing_game_ids)} existing games in database")

    # Initialize batch lists
    new_player_performances = []
    new_team_performances = []
    processed_count = 0
    error_count = 0

    for game_id in tqdm(list(game_ids), desc="Processing games"):
        # Skip if already in database
        if game_id in existing_game_ids:
            continue

        game_date = list(set(nba_games[nba_games.GAME_ID == game_id].GAME_DATE))[0]

        # Fetch box scores
        try:
            player_box_scores, team_box_scores = fetch_box_scores(game_id)
        except Exception as e:
            logger.warning(f"Failed to fetch box scores for game {game_id}: {e}")
            error_count += 1
            continue

        # Skip if empty
        if player_box_scores[0].empty or player_box_scores[1].empty:
            continue

        # Combine box scores
        try:
            combined_player = get_combined_player_box_score(*player_box_scores)
            combined_team = get_combined_team_box_score(*team_box_scores)
        except Exception as e:
            logger.warning(f"Failed to combine box scores for game {game_id}: {e}")
            error_count += 1
            continue

        if combined_player.empty or combined_team.empty:
            continue

        combined_player = combined_player.drop_duplicates()
        combined_team = combined_team.drop_duplicates()

        # Create player performance documents
        for _, row in combined_player.iterrows():
            try:
                performance = PlayerPerformance(
                    **row,
                    GAME_DATE=game_date,
                    PERCENTAGES=PlayerPercentages(**row),
                    ABSOLUTE_STATISTICS=AbsoluteStatistics(**row),
                    RATINGS=Ratings(**row),
                    MISC=Misc(**row),
                )
                # Check if not already in existing data
                existing = next(
                    (item for item in existing_player_performances
                     if item["GAME_ID"] == performance.GAME_ID
                     and item["PLAYER_ID"] == performance.PLAYER_ID),
                    None
                )
                if not existing:
                    new_player_performances.append(performance.model_dump())
            except Exception as e:
                logger.debug(f"Failed to create player performance: {e}")
                continue

        # Create team performance documents
        for _, row in combined_team.iterrows():
            try:
                performance = TeamPerformance(
                    **row,
                    GAME_DATE=game_date,
                    PERCENTAGES=TeamPercentages(**row),
                    ABSOLUTE_STATISTICS=AbsoluteStatistics(**row),
                    RATINGS=Ratings(**row),
                    MISC=Misc(**row),
                )
                existing = next(
                    (item for item in existing_team_performances
                     if item["GAME_ID"] == performance.GAME_ID
                     and item["TEAM_ID"] == performance.TEAM_ID),
                    None
                )
                if not existing:
                    new_team_performances.append(performance.model_dump())
            except Exception as e:
                logger.debug(f"Failed to create team performance: {e}")
                continue

        processed_count += 1

        # Upload batch when threshold reached
        if processed_count % UPLOAD_BATCH_SIZE == 0 and processed_count > 0:
            upload_batch(db, new_player_performances, new_team_performances)

            # Refresh existing data
            existing_player_performances = list(
                db.playerPerformances.find({}, projection=["GAME_ID", "PLAYER_ID"])
            )
            existing_team_performances = list(
                db.teamPerformances.find({}, projection=["GAME_ID", "TEAM_ID"])
            )
            existing_game_ids = set(pd.DataFrame(existing_team_performances).GAME_ID)

            # Reset batch lists
            new_player_performances = []
            new_team_performances = []

    # Upload remaining data
    if new_player_performances or new_team_performances:
        upload_batch(db, new_player_performances, new_team_performances)

    logger.info(f"Data collection complete. Processed {processed_count} games, {error_count} errors")


if __name__ == "__main__":
    collect_data()
