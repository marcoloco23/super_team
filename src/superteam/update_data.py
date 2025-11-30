"""
Data update module for fetching recent NBA game statistics.

This module fetches box score data from the NBA API and appends it to the existing
CSV data files. It's designed for incremental updates without requiring MongoDB.
"""

import time
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Set

from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoreadvancedv3,
    boxscoretraditionalv3,
    boxscorefourfactorsv3,
    boxscoremiscv3,
    boxscorescoringv3,
    boxscoreusagev3,
    boxscoreplayertrackv3,
)
import pandas as pd
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .logger import setup_logger
except ImportError:
    from superteam.logger import setup_logger

# Set up logging
logger = setup_logger("update_data")

# Configuration
API_RATE_LIMIT_DELAY = 0.6  # seconds between API calls
DATA_DIR = "data"
PLAYER_DATA_FILE = "player_data.csv"

# Column name mapping from v3 (camelCase) to v2 (UPPER_CASE) format
COLUMN_MAPPING = {
    'gameId': 'GAME_ID',
    'teamId': 'TEAM_ID',
    'teamCity': 'TEAM_CITY',
    'teamName': 'TEAM_NAME',
    'teamTricode': 'TEAM_ABBREVIATION',
    'teamSlug': 'TEAM_SLUG',
    'personId': 'PLAYER_ID',
    'firstName': 'FIRST_NAME',
    'familyName': 'FAMILY_NAME',
    'nameI': 'PLAYER_NAME',
    'playerSlug': 'PLAYER_SLUG',
    'position': 'START_POSITION',
    'comment': 'COMMENT',
    'jerseyNum': 'JERSEY_NUM',
    # Stats columns
    'minutes': 'MIN',
    'fieldGoalsMade': 'FGM',
    'fieldGoalsAttempted': 'FGA',
    'fieldGoalsPercentage': 'FG_PCT',
    'threePointersMade': 'FG3M',
    'threePointersAttempted': 'FG3A',
    'threePointersPercentage': 'FG3_PCT',
    'freeThrowsMade': 'FTM',
    'freeThrowsAttempted': 'FTA',
    'freeThrowsPercentage': 'FT_PCT',
    'reboundsOffensive': 'OREB',
    'reboundsDefensive': 'DREB',
    'reboundsTotal': 'REB',
    'assists': 'AST',
    'steals': 'STL',
    'blocks': 'BLK',
    'turnovers': 'TO',
    'foulsPersonal': 'PF',
    'points': 'PTS',
    'plusMinusPoints': 'PLUS_MINUS',
    # Advanced stats
    'estimatedOffensiveRating': 'E_OFF_RATING',
    'offensiveRating': 'OFF_RATING',
    'estimatedDefensiveRating': 'E_DEF_RATING',
    'defensiveRating': 'DEF_RATING',
    'estimatedNetRating': 'E_NET_RATING',
    'netRating': 'NET_RATING',
    'assistPercentage': 'AST_PCT',
    'assistToTurnover': 'AST_TOV',
    'assistRatio': 'AST_RATIO',
    'offensiveReboundPercentage': 'OREB_PCT',
    'defensiveReboundPercentage': 'DREB_PCT',
    'reboundPercentage': 'REB_PCT',
    'turnoverRatio': 'TM_TOV_PCT',
    'effectiveFieldGoalPercentage': 'EFG_PCT',
    'trueShootingPercentage': 'TS_PCT',
    'usagePercentage': 'USG_PCT',
    'estimatedUsagePercentage': 'E_USG_PCT',
    'estimatedPace': 'E_PACE',
    'pace': 'PACE',
    'pacePer40': 'PACE_PER40',
    'possessions': 'POSS',
    'pie': 'PIE',
}


def get_existing_game_ids(filepath: str) -> Set[str]:
    """Load existing game IDs from CSV file."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0)
        return set(df.GAME_ID.astype(str).unique())
    return set()


def fetch_games_since(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all NBA games since a given date.

    Args:
        start_date: Start date in MM/DD/YYYY format
        end_date: Optional end date in MM/DD/YYYY format

    Returns:
        DataFrame with game information
    """
    logger.info(f"Fetching games from {start_date} to {end_date or 'now'}...")

    gamefinder = leaguegamefinder.LeagueGameFinder(
        league_id_nullable="00",
        date_from_nullable=start_date,
        date_to_nullable=end_date,
    )
    games = gamefinder.get_data_frames()[0]
    logger.info(f"Found {len(games)} game records")
    return games


def fetch_box_score_safe(endpoint_class, game_id: str, max_retries: int = 3):
    """Fetch box score with retry logic."""
    for attempt in range(max_retries):
        try:
            time.sleep(API_RATE_LIMIT_DELAY)
            result = endpoint_class(game_id)
            return result.get_data_frames()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1} for game {game_id}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {endpoint_class.__name__} for {game_id}: {e}")
                return None
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert v3 column names to v2 format for consistency."""
    df = df.copy()
    df.columns = [COLUMN_MAPPING.get(c, c.upper()) for c in df.columns]
    return df


def fetch_player_box_scores(game_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch all box score types for a game and combine them.

    Args:
        game_id: NBA game ID

    Returns:
        Combined DataFrame with all player stats, or None if failed
    """
    endpoints = [
        boxscoreadvancedv3.BoxScoreAdvancedV3,
        boxscoretraditionalv3.BoxScoreTraditionalV3,
        boxscorefourfactorsv3.BoxScoreFourFactorsV3,
        boxscoremiscv3.BoxScoreMiscV3,
        boxscorescoringv3.BoxScoreScoringV3,
        boxscoreusagev3.BoxScoreUsageV3,
        boxscoreplayertrackv3.BoxScorePlayerTrackV3,
    ]

    dfs = []
    merge_cols = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']

    for endpoint_class in endpoints:
        result = fetch_box_score_safe(endpoint_class, game_id)
        if result is None or len(result) == 0:
            return None

        df = result[0]  # Player stats are in first dataframe
        if df.empty:
            return None

        # Standardize column names
        df = standardize_columns(df)
        dfs.append(df)

    # Merge all dataframes
    combined = dfs[0]
    for df in dfs[1:]:
        # Get columns that exist in both for merging
        common_merge_cols = [c for c in merge_cols if c in df.columns and c in combined.columns]
        # Get new columns (not already in combined)
        new_cols = [c for c in df.columns if c not in combined.columns]
        if new_cols and common_merge_cols:
            combined = combined.merge(
                df[common_merge_cols + new_cols],
                on=common_merge_cols,
                how='outer'
            )

    # Create PLAYER_NAME from first/last name if not present
    if 'PLAYER_NAME' not in combined.columns and 'FIRST_NAME' in combined.columns:
        combined['PLAYER_NAME'] = combined['FIRST_NAME'] + ' ' + combined.get('FAMILY_NAME', '')

    # Add NICKNAME column if missing (required by existing code)
    if 'NICKNAME' not in combined.columns:
        combined['NICKNAME'] = ''

    return combined


def update_player_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_games: Optional[int] = None
) -> pd.DataFrame:
    """
    Update player data with recent games.

    Args:
        start_date: Start date (defaults to last date in existing data + 1 day)
        end_date: End date (defaults to today)
        max_games: Maximum number of new games to fetch (for testing)

    Returns:
        DataFrame with new player data
    """
    filepath = os.path.join(DATA_DIR, PLAYER_DATA_FILE)

    # Get existing game IDs
    existing_ids = get_existing_game_ids(filepath)
    logger.info(f"Found {len(existing_ids)} existing games in {filepath}")

    # Determine date range
    if start_date is None:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0)
            last_date = pd.to_datetime(df.GAME_DATE).max()
            start_date = (last_date + timedelta(days=1)).strftime("%m/%d/%Y")
        else:
            # Default to 1 year ago
            start_date = (datetime.now() - timedelta(days=365)).strftime("%m/%d/%Y")

    if end_date is None:
        end_date = datetime.now().strftime("%m/%d/%Y")

    # Fetch games
    games_df = fetch_games_since(start_date, end_date)
    if games_df.empty:
        logger.info("No new games found")
        return pd.DataFrame()

    # Get unique game IDs that we don't have yet
    new_game_ids = set(games_df.GAME_ID.astype(str)) - existing_ids
    logger.info(f"Found {len(new_game_ids)} new games to fetch")

    if max_games:
        new_game_ids = list(new_game_ids)[:max_games]

    # Create game_id to date mapping
    game_dates = games_df.drop_duplicates('GAME_ID').set_index('GAME_ID')['GAME_DATE'].to_dict()

    # Fetch box scores for new games
    new_data = []
    failed_games = []

    for game_id in tqdm(list(new_game_ids), desc="Fetching box scores"):
        player_data = fetch_player_box_scores(game_id)
        if player_data is not None and not player_data.empty:
            player_data['GAME_DATE'] = game_dates.get(game_id, game_dates.get(int(game_id), ''))
            new_data.append(player_data)
        else:
            failed_games.append(game_id)

    if failed_games:
        logger.warning(f"Failed to fetch {len(failed_games)} games")

    if not new_data:
        logger.info("No new data fetched")
        return pd.DataFrame()

    # Combine new data
    new_df = pd.concat(new_data, ignore_index=True)
    logger.info(f"Fetched {len(new_df)} new player records from {len(new_data)} games")

    # Append to existing file
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath, index_col=0)

        # Align columns - add missing columns with NaN
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        for col in new_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = None

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID'])
    else:
        combined_df = new_df

    # Save
    combined_df.to_csv(filepath)
    logger.info(f"Saved {len(combined_df)} total records to {filepath}")

    return new_df


def main():
    """Main update function."""
    import argparse

    parser = argparse.ArgumentParser(description="Update NBA player data")
    parser.add_argument("--start-date", help="Start date (MM/DD/YYYY)")
    parser.add_argument("--end-date", help="End date (MM/DD/YYYY)")
    parser.add_argument("--max-games", type=int, help="Max games to fetch")

    args = parser.parse_args()

    update_player_data(
        start_date=args.start_date,
        end_date=args.end_date,
        max_games=args.max_games
    )


if __name__ == "__main__":
    main()
