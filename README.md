# Superteam

A basketball analytics framework that uses machine learning to predict team performance and identify competitive NBA team compositions.

## Overview

Superteam uses XGBoost regression to predict team performance metrics (plus-minus scores) based on collective player statistics from NBA games. The system can:

- Simulate matchups between any two NBA teams
- Simulate an entire regular season to rank all 30 teams
- Run tournament-style brackets with random team compositions
- Find optimal team compositions within salary cap constraints
- Suggest trades to improve team performance
- Build optimal teams around a specific player

## Features

- **Data Collection**: Fetches comprehensive box score data from 7 different NBA API endpoints
- **Machine Learning**: XGBoost models trained on 10,000+ games
- **Interactive Dashboard**: Streamlit web application for exploration
- **Trade Analysis**: Find value-matched trades to improve team performance
- **Salary Cap Awareness**: Optional salary cap constraints for realistic team building

## Installation

### Prerequisites

- Python 3.9+
- MongoDB Atlas account (for data storage)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd super_team
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package:
   ```bash
   pip install -e ".[dev]"
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB credentials
   ```

## Usage

### Web Application

Run the Streamlit dashboard:

```bash
streamlit run src/superteam/app.py
```

The dashboard provides the following applications:

1. **Raw Data**: Browse player statistics with custom scoring metric
2. **Simulate Matchup**: Compare any two NBA teams
3. **Simulate Regular Season**: Rank all 30 NBA teams
4. **Simulate Tournament**: Run multi-round tournament brackets
5. **Build Team Around Player**: Optimize roster with a specific player
6. **Get Super Team**: Find the best team composition
7. **Trade Finder**: Suggest roster improvements via trades

### Data Collection

To collect fresh data from the NBA API:

```bash
python src/superteam/collect_data.py
```

This process:
- Fetches game data from multiple NBA API endpoints
- Stores player and team performance statistics in MongoDB
- Implements rate limiting to respect API constraints

### Model Training

To train new models:

```bash
python src/superteam/model.py
```

The training script:
- Loads game data from MongoDB
- Preprocesses features for matchup prediction
- Trains XGBoost regression models
- Saves models for different team sizes (1, 5, 8, 10, 13 players)

## Project Structure

```
super_team/
├── src/
│   └── superteam/          # Main package
│       ├── __init__.py     # Package initialization
│       ├── app.py          # Streamlit web application
│       ├── simulation.py   # Team simulation & optimization
│       ├── helpers.py      # Utility functions
│       ├── model.py        # Model training script
│       ├── collect_data.py # Data collection from NBA API
│       ├── models.py       # Pydantic data models
│       ├── constants.py    # Configuration (environment variables)
│       └── logger.py       # Logging configuration
├── tests/                  # Test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── test_helpers.py     # Helper function tests
│   ├── test_simulation.py  # Simulation function tests
│   └── test_models.py      # Pydantic model tests
├── notebooks/              # Jupyter notebooks
├── data/                   # Player data CSVs
├── models/                 # Trained XGBoost models
│   ├── 1_player_model.json
│   ├── 5_player_model.json
│   ├── 8_player_model.json
│   ├── 10_player_model.json
│   └── 13_player_model.json
├── pyproject.toml          # Python packaging config
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment file
└── doc/                    # Documentation
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_PW` | MongoDB password | (required) |
| `MONGO_DB` | MongoDB database name | `dev` |
| `MONGO_NAME` | MongoDB username | `superteam` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FILE` | Log file path (optional) | (none) |

### Model Configuration

The model uses the following hyperparameters:

- Booster: `gbtree`
- Learning rate: `0.1`
- Estimators: `100`
- Max depth: `4`
- Early stopping: `50` rounds

## How It Works

### Data Pipeline

1. **Collection**: Box scores are fetched from 7 NBA API endpoints (advanced stats, tracking, traditional, four factors, misc, scoring, usage)
2. **Storage**: Raw data is stored in MongoDB collections
3. **Preprocessing**: Statistics are flattened and normalized for model input
4. **Training**: XGBoost models are trained on historical matchup data
5. **Prediction**: Models predict plus-minus for team matchups

### Simulation Algorithm

1. For each team, get the top N players by minutes played
2. Create feature vectors from player statistics
3. Stack features for both teams into a single input
4. Model predicts plus-minus differential
5. Team with higher prediction wins

### Team Optimization

The `get_super_team` function uses a genetic algorithm-style approach:

1. Start with a random team
2. Generate random challenger teams
3. If challenger beats current best (and meets salary cap), become new best
4. Repeat for specified iterations
5. Return best team found

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=superteam --cov-report=html
```

The test suite includes:
- Unit tests for helper functions
- Unit tests for simulation functions
- Unit tests for Pydantic data models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

- NBA API for providing comprehensive basketball statistics
- XGBoost for the machine learning framework
- Streamlit for the interactive dashboard
