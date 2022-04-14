Superteam is a basketball analytics framework for identifying strong competitive teams from individual player performances. By collecting large amounts of detailed player performance data over 15000 NBA games, an XGB regression model is trained to predict a teams plus minus score for a specific game from the corresponding collective player performances of both teams.

As the outcomes of this model are relative with respect to an opposing team, an elimination tournament is set up similar to the NBA playoffs in which the winner has to go unbeaten throughout all rounds (instead of playing a best of 7 series a single simulated match will be used to determine the winner). By randomly selecting players from a player pool and calculating their average performance statistics, teams are created and matched up against each other until a winner emerges.

Individual teams are can also be matched up against a large number of randomly selected teams and win loss outcome statistics can be calculated in a way similar to the regular NBA season. By iteratively finding better teams, we eventually identify teams which remain undefeated against a large number of randomly selected opponents.

As of now, we require a fixed amount of players for each team per game which we have set to the official NBA limit of 13 players. A model, requiring only the starting 5 players is also provided.

We currently have no clear way of identifying why a team is better than another team other than by explicity looking at the feature importances of the trained model.

As of now over 500 NBA players are included with performance data dating back to 2015.

To get started, simply run a tournament, simulate a matchup, or find a super team using the simulation jupyter notebook.

All data is fetched using the python nba_api and is stored using MongoDB.