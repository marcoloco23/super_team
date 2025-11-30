import streamlit as st
import pandas as pd
import xgboost as xgb
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta

# Handle both relative and absolute imports for Streamlit compatibility
try:
    from .helpers import (
        get_average_player_performances,
        get_score_df,
    )
    from .simulation import (
        build_team_around_player,
        get_super_team,
        nba_trade_finder,
        run_tournament,
        simulate_arbitrary_matchup,
        simulate_nba_matchup,
        simulate_regular_season,
        evaluate_team,
    )
except ImportError:
    from superteam.helpers import (
        get_average_player_performances,
        get_score_df,
    )
    from superteam.simulation import (
        build_team_around_player,
        get_super_team,
        nba_trade_finder,
        run_tournament,
        simulate_arbitrary_matchup,
        simulate_nba_matchup,
        simulate_regular_season,
        evaluate_team,
    )

gamefinder = leaguegamefinder.LeagueGameFinder()


@st.cache_data
def load_data():
    data = pd.read_csv("data/player_data.csv", index_col=0, low_memory=False)
    # Convert MIN to numeric (stored as string in some rows)
    if 'MIN' in data.columns and data['MIN'].dtype == object:
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
        data['MIN'] = data['MIN'].apply(parse_min)
    return data


@st.cache_data
def get_teams_df():
    return pd.DataFrame(teams.get_teams())


@st.cache_data
def get_player_team_dict(_data):
    """Get a mapping of player IDs to team abbreviations from local data."""
    # Sort by game date to get most recent team for each player
    sorted_data = _data.sort_values('GAME_DATE')
    # Get the last (most recent) team for each player
    return sorted_data.groupby('PLAYER_ID')['TEAM_ABBREVIATION'].last().to_dict()


def insert_team_abbreviation(average_performances, player_team_dict):
    average_performances["TEAM_ABBREVIATION"] = average_performances.PLAYER_ID.map(
        player_team_dict
    )
    average_performances = average_performances.dropna()
    first_column = average_performances.pop("TEAM_ABBREVIATION")
    average_performances.insert(0, "TEAM_ABBREVIATION", first_column)
    return average_performances


st.title("Super Team")

# Load Data
data = load_data()
teams_df = get_teams_df()
player_team_dict = get_player_team_dict(data)


# Set Options
applications = [
    "Raw Data",
    "Simulate Matchup",
    "Simulate Regular Season",
    "Simulate Tournament",
    "Build Team Around Player",
    "Get Super Team",
    "Trade Finder",
]
app = st.sidebar.selectbox("Application", applications)

# Filter Data
start_date = st.sidebar.date_input(
    "Data start date", datetime.datetime.now() - relativedelta(years=1)
).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input("Data end Date").strftime("%Y-%m-%d")

filtered_data = data[data.GAME_DATE <= end_date]
filtered_data = filtered_data[filtered_data.GAME_DATE >= start_date]
average_performances = insert_team_abbreviation(
    get_average_player_performances(filtered_data), player_team_dict
)
average_performances = average_performances.drop(["GAME_ID"], axis=1)
player_names = average_performances.PLAYER_NAME
mapping_player_id_name = dict(zip(average_performances.PLAYER_ID, player_names))
team_list = teams_df.full_name
mapping_full_name_abbrev = dict(zip(teams_df.full_name, teams_df.abbreviation))
trade_value_df = get_score_df(average_performances)

# Apps
if app == applications[0]:
    st.subheader("Raw data")
    st.write(average_performances)

    st.subheader("Custom All in 1 Metric")
    st.write(trade_value_df)

# Load Model
if app != applications[0]:
    team_size = st.sidebar.selectbox("Team Size (players)", (1, 5, 8, 10, 13), 2)
    model = xgb.XGBRegressor()
    model.load_model(f"models/{team_size}_player_model.json")

# Simulate Matchup
if app == applications[1]:

    team_A = st.sidebar.selectbox("Team A?", team_list, 0)
    team_B = st.sidebar.selectbox("Team B?", team_list, 1)

    team_A_abbrev = mapping_full_name_abbrev.get(team_A)
    team_B_abbrev = mapping_full_name_abbrev.get(team_B)

    score = simulate_nba_matchup(
        team_A_abbrev,
        team_B_abbrev,
        average_performances,
        model=model,
        team_size=team_size,
    )
    if score[0] > score[1]:
        result = f"{team_A} wins"
    if score[1] > score[0]:
        result = f"{team_B} wins"
    if score[1] == score[0]:
        result = "Tie"

    st.subheader(applications[1])
    st.write(pd.DataFrame(score, index=[team_A, team_B], columns=["Predicted +/-"]))
    st.write(result)

# Simulate Regular Season
if app == applications[2]:
    st.subheader(applications[2])
    with st.spinner("Simulating Regular Season..."):
        regular_season_results = simulate_regular_season(
            average_performances, model=model, teams_df=teams_df, team_size=team_size
        )
    st.success("Done!")
    st.write(pd.DataFrame(regular_season_results, index=["Rankings"]).T)

# Simulate Tournament
if app == applications[3]:
    st.subheader(applications[3])
    rounds = st.sidebar.select_slider("Tournament Rounds", range(1, 101))
    with st.spinner("Running Tournaments..."):
        winner_name_list, winner_id_list = run_tournament(
            average_performances,
            model=model,
            rounds=rounds,
            team_count=32,
            team_size=team_size,
        )
    st.success("Done!")
    st.subheader("Winning Teams")
    st.write(pd.DataFrame(winner_name_list))


if app == applications[4]:
    st.subheader(applications[4])
    iterations = st.sidebar.select_slider("Iterations", range(10, 1000))
    salary_cap = st.sidebar.checkbox("Salary Cap", True)
    player_name = st.selectbox("Select Player", player_names.to_list())
    with st.spinner(f"Building Team around {player_name}..."):
        player_ids = build_team_around_player(
            player_name,
            average_performances,
            model=model,
            team_size=team_size,
            iterations=iterations,
            salary_cap=salary_cap,
        )
        score = evaluate_team(
            player_ids,
            average_performances,
            model=model,
            team_size=team_size,
            iterations=100,
        )
    team_data = average_performances[
        average_performances.PLAYER_ID.isin(player_ids)
    ].sort_values("MIN", ascending=False)
    st.success("Done!")
    st.write(team_data)
    st.write(f"Win Loss Ratio: {score} after simulating 100 games")

if app == applications[5]:
    st.subheader(applications[5])
    iterations = st.sidebar.select_slider("Iterations", range(100, 1000))
    salary_cap = st.sidebar.checkbox("Salary Cap", True)
    with st.spinner(f"Getting Super Team..."):
        super_team_ids = get_super_team(
            average_performances,
            model=model,
            team_size=team_size,
            iterations=iterations,
            salary_cap=salary_cap,
        )
        score = evaluate_team(
            super_team_ids,
            average_performances,
            model=model,
            team_size=team_size,
            iterations=100,
        )
    st.success("Done!")
    st.write(
        average_performances[
            average_performances.PLAYER_ID.isin(super_team_ids)
        ].sort_values("MIN", ascending=False)
    )
    st.write(f"Win Loss Ratio: {score} after simulating 100 games")

if app == applications[6]:
    st.subheader(applications[6])
    team = st.sidebar.selectbox("Team?", team_list, 0)
    samples = st.sidebar.select_slider("Samples", range(1, 100))
    team_abbrev = mapping_full_name_abbrev.get(team)
    with st.spinner(f"Identifying Possible Trades..."):
        (
            traded_player_name,
            acquired_player_name,
            base_score,
            best_score,
        ) = nba_trade_finder(
            team_abbrev,
            average_performances,
            model=model,
            team_size=team_size,
            samples=samples,
        )
    st.success("Done!")
    if base_score >= best_score:
        st.write("No Improvements Found")
    else:
        st.write(
            f"Trade {traded_player_name} for {acquired_player_name} to improve from {round(base_score,3)} to {round(best_score,3)} W/L"
        )
