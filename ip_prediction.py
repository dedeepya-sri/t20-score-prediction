import pandas as pd  
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder  

# Load datasets
matches = pd.read_csv(r"C:\Users\dedee\OneDrive\Desktop\ipl\matches.csv")  
deliveries = pd.read_csv(r"C:\Users\dedee\OneDrive\Desktop\ipl\deliveries.csv")  

# Data Preprocessing
matches = matches[['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner', 
                   'toss_decision', 'result', 'winner']]

deliveries = deliveries[['match_id', 'inning', 'batting_team', 'bowling_team', 'over', 
                         'ball', 'batter', 'bowler', 'total_runs', 'is_wicket']]

# Merging data
match_runs = deliveries.groupby('match_id')['total_runs'].sum().reset_index()
match_runs.rename(columns={'total_runs': 'match_total_runs'}, inplace=True)
matches = matches.merge(match_runs, left_on='id', right_on='match_id', how='left')

avg_runs_per_over = deliveries.groupby(['match_id', 'over'])['total_runs'].sum().reset_index()
avg_runs_per_over = avg_runs_per_over.groupby('match_id')['total_runs'].mean().reset_index()
avg_runs_per_over.rename(columns={'total_runs': 'avg_runs_per_over'}, inplace=True)
matches = matches.merge(avg_runs_per_over, on='match_id', how='left')

# Counting matches played and won
team_matches = pd.concat([matches['team1'], matches['team2']]).value_counts().reset_index()
team_matches.columns = ['team', 'total_matches']

team_wins = matches['winner'].value_counts().reset_index()
team_wins.columns = ['team', 'wins']

# Merge team stats
team_stats = pd.merge(team_matches, team_wins, on='team', how='left')
team_stats['win_percentage'] = (team_stats['wins'] / team_stats['total_matches']) * 100
team_stats.fillna(0, inplace=True)

# Powerplay and death overs runs
powerplay_runs = deliveries[deliveries['over'] <= 6].groupby('match_id')['total_runs'].sum().reset_index()
powerplay_runs.rename(columns={'total_runs': 'powerplay_runs'}, inplace=True)

death_over_runs = deliveries[deliveries['over'] >= 16].groupby('match_id')['total_runs'].sum().reset_index()
death_over_runs.rename(columns={'total_runs': 'death_overs_runs'}, inplace=True)

# Merging powerplay and death over runs with matches dataset
matches = matches.merge(powerplay_runs, on='match_id', how='left')
matches = matches.merge(death_over_runs, on='match_id', how='left')

# Venue average runs
venue_avg_runs = matches.groupby('city')['match_total_runs'].mean().reset_index()
venue_avg_runs.rename(columns={'match_total_runs': 'avg_runs_at_venue'}, inplace=True)
matches = matches.merge(venue_avg_runs, on='city', how='left')

# Encoding categorical variables
le = LabelEncoder()
matches['team1'] = le.fit_transform(matches['team1'])
matches['team2'] = le.fit_transform(matches['team2'])
matches['toss_winner'] = le.fit_transform(matches['toss_winner'])
matches['winner'] = le.fit_transform(matches['winner'])
matches['city'] = le.fit_transform(matches['city'])
matches['toss_decision'] = le.fit_transform(matches['toss_decision'])

# Defining features and target variable
X = matches[['team1', 'team2', 'toss_winner', 'toss_decision', 'city', 'powerplay_runs', 'death_overs_runs']]
y = matches['match_total_runs']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values with 0 and convert to numeric types
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5
print(f"Linear Regression - MAE: {mae_lr}, RMSE: {rmse_lr}")

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
print(f"Random Forest - MAE: {mae_rf}, RMSE: {rmse_rf}")

# Grid Search for Hyperparameter Tuning (Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

y_pred_best_rf = best_rf_model.predict(X_test)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
rmse_best_rf = mean_squared_error(y_test, y_pred_best_rf) ** 0.5
print(f"Best Random Forest - MAE: {mae_best_rf}, RMSE: {rmse_best_rf}")

# XGBoost Model
xg_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xg_model.fit(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)
mae_xg = mean_absolute_error(y_test, y_pred_xg)
rmse_xg = mean_squared_error(y_test, y_pred_xg) ** 0.5
print(f"XGBoost - MAE: {mae_xg}, RMSE: {rmse_xg}")

# Save the best model (Random Forest)
joblib.dump(best_rf_model, 'ipl_scorepred.joblib')

# Load the saved model
loaded_model = joblib.load('ipl_scorepred.joblib')

# Example: Predict score for a new match
new_match = [[1, 2, 1, 1, 3, 45, 32]]  # Example data for team1, team2, toss_winner, etc.
predicted_score = loaded_model.predict(new_match)
print(f"Predicted Score: {predicted_score[0]}")

