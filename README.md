Superteam is a basketball analytics framework for identifying strong competitive teams from individual player performances. By collecting large amounts of detailed player performance data over 15000 NBA games, a regression model is trained to predict the plus minus score for a specific game from the corresponding collective player performances of both teams. New teams can then be created by combining average player performances over a given period of time.

As the outcomes of this model are relative with respect to an opposing team, an elimination tournament is set up similar to the NBA playoff in which the winner has to go unbeaten throughout all round (instead of playing a best of 7 series a single simulated match will be used to determine the winner). By randomly selecting players from a player pool and calculating their average performance statistics, teams are created and matched up against each other.

Individual teams are can also be matched up against a large number of randomly selected teams and win loss outcome statistics can be calculated in a way similar to the regular NBA season. By iteratively finding better teams, we eventually identify teams which remain undefeated against a large number of randomly selected opponents.

As of now, we require a fixed amount of players for each team per game which we have set to the official NBA limit of 13 players. A model, requiring only the starting 5 players is also provided.

We currently have no clear way of identifying why a team is better than another team other than by explicity looking at the feature importances of the trained model. 

To get started, simply run a tournament and view the outcome.