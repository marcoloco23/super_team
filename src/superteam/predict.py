"""
Easy NBA Game Prediction Module for Super Team.

Simple functions to predict NBA game outcomes with estimated final scores.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Import from existing modules
try:
    from .helpers import get_average_player_performances
    from .simulation import simulate_nba_matchup
except ImportError:
    from superteam.helpers import get_average_player_performances
    from superteam.simulation import simulate_nba_matchup


# NBA team mappings
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Average NBA game score
AVG_TEAM_SCORE = 112


def _convert_min_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MIN column to numeric, handling various formats."""
    if 'MIN' not in df.columns:
        return df

    df = df.copy()
    if df['MIN'].dtype == object:
        def parse_min(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val)
            if ':' in val:
                parts = val.split(':')
                try:
                    return float(parts[0]) + float(parts[1]) / 60
                except (ValueError, IndexError):
                    return np.nan
            try:
                return float(val)
            except ValueError:
                return np.nan
        df['MIN'] = df['MIN'].apply(parse_min)
    else:
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')

    return df


class GamePredictor:
    """Easy-to-use NBA game predictor."""

    def __init__(self, data_path: str = "data/player_data.csv",
                 model_path: str = "models/8_player_model.json",
                 team_size: int = 8):
        """
        Initialize the predictor.

        Args:
            data_path: Path to player data CSV
            model_path: Path to trained XGBoost model
            team_size: Number of players per team (must match model)
        """
        self.team_size = team_size
        print("Loading player data...")
        self.data = pd.read_csv(data_path, index_col=0, low_memory=False)

        # Convert MIN to numeric
        self.data = _convert_min_to_numeric(self.data)

        print("Calculating player averages...")
        self.avg_perf = get_average_player_performances(self.data)

        # Ensure MIN is in the averages
        if 'MIN' not in self.avg_perf.columns:
            min_avg = self.data.groupby('PLAYER_ID')['MIN'].mean()
            self.avg_perf['MIN'] = self.avg_perf['PLAYER_ID'].map(min_avg)

        # Add team abbreviations from most recent games
        sorted_data = self.data.sort_values("GAME_DATE")
        latest_teams = sorted_data.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last()
        self.avg_perf["TEAM_ABBREVIATION"] = self.avg_perf["PLAYER_ID"].map(latest_teams)
        self.avg_perf = self.avg_perf.dropna(subset=["TEAM_ABBREVIATION"])

        # Filter to only NBA teams
        nba_team_abbrevs = set(NBA_TEAMS.keys())
        self.avg_perf = self.avg_perf[self.avg_perf['TEAM_ABBREVIATION'].isin(nba_team_abbrevs)]

        # Move TEAM_ABBREVIATION after PLAYER_NAME
        cols = self.avg_perf.columns.tolist()
        if "TEAM_ABBREVIATION" in cols:
            cols.remove("TEAM_ABBREVIATION")
            cols.insert(2, "TEAM_ABBREVIATION")
            self.avg_perf = self.avg_perf[cols]

        print("Loading model...")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        print(f"Ready! ({len(self.avg_perf)} players loaded)")

    def predict_game(self, home_team: str, away_team: str) -> Dict:
        """
        Predict the outcome of a game.

        Args:
            home_team: Home team abbreviation (e.g., 'LAL', 'BOS')
            away_team: Away team abbreviation

        Returns:
            Dictionary with prediction results
        """
        home_team = home_team.upper()
        away_team = away_team.upper()

        if home_team not in NBA_TEAMS:
            raise ValueError(f"Unknown team: {home_team}. Valid: {list(NBA_TEAMS.keys())}")
        if away_team not in NBA_TEAMS:
            raise ValueError(f"Unknown team: {away_team}. Valid: {list(NBA_TEAMS.keys())}")

        # Use simulation.py's function - this ensures consistent feature processing
        plus_minus = simulate_nba_matchup(
            home_team, away_team,
            self.avg_perf,
            model=self.model,
            team_size=self.team_size
        )

        home_pm = plus_minus[0]
        away_pm = plus_minus[1]

        # Calculate scores with home court advantage (~3 points)
        home_advantage = 3
        point_diff = (home_pm - away_pm) / 2 + home_advantage

        home_score = round(AVG_TEAM_SCORE + point_diff / 2)
        away_score = round(AVG_TEAM_SCORE - point_diff / 2)

        # Win probability (sigmoid)
        win_prob = 1 / (1 + np.exp(-point_diff / 5))

        winner = home_team if home_score > away_score else away_team

        return {
            'home_team': home_team,
            'home_team_name': NBA_TEAMS[home_team],
            'away_team': away_team,
            'away_team_name': NBA_TEAMS[away_team],
            'home_score': home_score,
            'away_score': away_score,
            'point_spread': round(point_diff, 1),
            'home_win_probability': round(win_prob * 100, 1),
            'away_win_probability': round((1 - win_prob) * 100, 1),
            'predicted_winner': winner,
            'predicted_winner_name': NBA_TEAMS[winner],
        }

    def predict_season(self) -> pd.DataFrame:
        """Predict full season standings."""
        results = []
        team_list = list(NBA_TEAMS.keys())

        print("Simulating season...")
        for i, team in enumerate(team_list):
            wins = 0
            losses = 0
            point_diff_total = 0

            for opponent in team_list:
                if team == opponent:
                    continue

                try:
                    # Home game
                    pred = self.predict_game(team, opponent)
                    if pred['predicted_winner'] == team:
                        wins += 1
                    else:
                        losses += 1
                    point_diff_total += pred['point_spread']

                    # Away game
                    pred = self.predict_game(opponent, team)
                    if pred['predicted_winner'] == team:
                        wins += 1
                    else:
                        losses += 1
                    point_diff_total -= pred['point_spread']
                except Exception:
                    continue

            total_games = wins + losses
            if total_games > 0:
                win_pct = wins / total_games
                projected_wins = round(win_pct * 82)
                projected_losses = 82 - projected_wins

                results.append({
                    'Team': team,
                    'Team Name': NBA_TEAMS[team],
                    'Projected Wins': projected_wins,
                    'Projected Losses': projected_losses,
                    'Win %': round(win_pct * 100, 1),
                    'Avg Point Diff': round(point_diff_total / total_games, 1)
                })

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(team_list)} teams processed...")

        df = pd.DataFrame(results)
        return df.sort_values('Projected Wins', ascending=False).reset_index(drop=True)

    def get_team_roster(self, team_abbrev: str, top_n: int = 10) -> pd.DataFrame:
        """Get top players for a team."""
        team_abbrev = team_abbrev.upper()
        players = self.avg_perf[
            self.avg_perf.TEAM_ABBREVIATION == team_abbrev
        ].sort_values("MIN", ascending=False).head(top_n)

        cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION']
        for c in ['PTS', 'AST', 'REB', 'MIN']:
            if c in players.columns:
                cols.append(c)

        return players[cols].round(1)


# Singleton predictor
_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = GamePredictor()
    return _predictor


def predict(home: str, away: str) -> Dict:
    """Quick game prediction."""
    return _get_predictor().predict_game(home, away)


def matchup(home: str, away: str) -> str:
    """Formatted matchup prediction."""
    p = predict(home, away)
    return f"""
{p['home_team_name']} vs {p['away_team_name']}
{'='*40}
Predicted Score: {p['home_team']} {p['home_score']} - {p['away_team']} {p['away_score']}
Point Spread: {p['home_team']} {p['point_spread']:+.1f}
Winner: {p['predicted_winner_name']} ({max(p['home_win_probability'], p['away_win_probability']):.0f}% confidence)
"""


def season_preview() -> pd.DataFrame:
    """Get season predictions for all teams."""
    return _get_predictor().predict_season()


def roster(team: str) -> pd.DataFrame:
    """Get team roster with stats."""
    return _get_predictor().get_team_roster(team)


def teams_list() -> List[str]:
    """Get list of valid team abbreviations."""
    return list(NBA_TEAMS.keys())


def main():
    """Interactive prediction CLI."""
    print("\n" + "="*50)
    print("NBA GAME PREDICTOR")
    print("="*50 + "\n")

    predictor = GamePredictor()

    print("\nAvailable teams:")
    for i, (abbrev, name) in enumerate(sorted(NBA_TEAMS.items())):
        print(f"  {abbrev}: {name}", end="")
        if (i + 1) % 3 == 0:
            print()
    print("\n")

    while True:
        print("\nCommands:")
        print("  TEAM vs TEAM  - Predict a game (e.g., 'LAL vs BOS')")
        print("  season        - Full season predictions")
        print("  roster TEAM   - Team roster (e.g., 'roster LAL')")
        print("  q             - Quit")

        user_input = input("\n> ").strip().lower()

        if user_input in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break

        if user_input == 'season':
            standings = predictor.predict_season()
            print("\nPROJECTED STANDINGS\n")
            print(standings.to_string(index=False))
            continue

        if user_input.startswith('roster '):
            team = user_input.replace('roster ', '').upper()
            if team in NBA_TEAMS:
                print(f"\n{NBA_TEAMS[team]} Roster:\n")
                print(predictor.get_team_roster(team).to_string(index=False))
            else:
                print(f"Unknown team: {team}")
            continue

        # Parse matchup
        parts = user_input.replace(' vs ', ' ').replace('@', ' ').split()
        if len(parts) >= 2:
            home = parts[0].upper()
            away = parts[1].upper()

            if home in NBA_TEAMS and away in NBA_TEAMS:
                try:
                    result = predictor.predict_game(home, away)
                    print(f"\n{'='*45}")
                    print(f"{result['home_team_name']} vs {result['away_team_name']}")
                    print(f"{'='*45}")
                    print(f"\nPREDICTED SCORE")
                    print(f"  {result['home_team']}: {result['home_score']}")
                    print(f"  {result['away_team']}: {result['away_score']}")
                    print(f"\nWINNER: {result['predicted_winner_name']}")
                    print(f"  Win Probability: {max(result['home_win_probability'], result['away_win_probability']):.1f}%")
                    print(f"  Point Spread: {result['home_team']} {result['point_spread']:+.1f}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print(f"Unknown team(s). Valid: {', '.join(sorted(NBA_TEAMS.keys()))}")
        else:
            print("Enter: TEAM vs TEAM (e.g., 'LAL vs BOS')")


if __name__ == "__main__":
    main()
