import streamlit as st
import pandas as pd
import xgboost as xgb
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from helpers import (
    get_average_player_performances,
    get_score_df,
    get_team_player_ids,
    insert_team_abbreviation,
)
from simulation import (
    build_team_around_player,
    get_super_team,
    nba_trade_finder,
    run_tournament,
    simulate_arbitrary_matchup,
    simulate_nba_matchup,
    simulate_regular_season,
    test_team,
)
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta

gamefinder = leaguegamefinder.LeagueGameFinder()


@st.cache
def load_data():
    data = pd.read_csv("data/player_data.csv", index_col=0)
    return data


@st.cache
def get_teams_df():
    return pd.DataFrame(teams.get_teams())


@st.cache
def get_player_team_dict(teams_df):
    team_abbreviations = teams_df.abbreviation.to_list()
    player_team_dict = {}
    for team_abb in team_abbreviations:
        player_ids = get_team_player_ids(team_abb)
        for player_id in player_ids:
            player_team_dict[player_id] = team_abb
    return player_team_dict


def simulate_regular_season(average_performances, model, teams_df, team_size=13):
    team_abbreviations = teams_df.abbreviation.to_list()
    results_dict = {}
    for i, team_A in enumerate(team_abbreviations):
        win_loss_list = []
        for team_B in [*team_abbreviations[:i], *team_abbreviations[i + 1 :]]:
            plus_minus_prediction = simulate_nba_matchup(
                team_A, team_B, average_performances, model=model, team_size=team_size
            )
            if plus_minus_prediction[0] > plus_minus_prediction[1]:
                win_loss_list.append(1)
            else:
                win_loss_list.append(0)

        results_dict[team_A] = np.mean(win_loss_list)
    results = dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))
    return results


def run_tournament(average_performances, model, rounds=1, team_count=16, team_size=13):
    """Team Count must be a power of 2"""
    winner = False
    winner_name_list = []
    winner_id_list = []
    for _ in range(rounds):
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
            for teamA, teamB in zip(it, it):
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
            winner = True
            winner_name_list.append(
                winner_team.sort_values("MIN", ascending=False).PLAYER_NAME.to_list()
            )
            winner_id_list.append(
                winner_team.sort_values("MIN", ascending=False).PLAYER_ID.to_list()
            )
    return winner_name_list, winner_id_list


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
player_team_dict = get_player_team_dict(teams_df)


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
        score = test_team(
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
        score = test_team(
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
