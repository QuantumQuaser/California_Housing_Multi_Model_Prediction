# -*- coding: utf-8 -*-
"""california_data_set_multi model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-qguqjfte3I0_jJqdbGm_Lden_ml-8p7
"""



# Importing necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
california_data = california_housing.frame

california_data.describe()
california_data.info()

california_data.hist(bins=50, figsize=(20,15))

import matplotlib.pyplot as plt
california_data.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,
                     s=california_data["Population"]/100, label="Population",
                     figsize=(10,7), c="MedHouseVal", cmap=plt.get_cmap("jet"), colorbar=True)

import seaborn as sns
correlation_matrix = california_data.corr()
sns.heatmap(correlation_matrix, annot=True)

sns.pairplot(california_data)

california_data.boxplot(figsize=(20,15))

# Importing necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Fetch the California Housing dataset
california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target

# Convert to DataFrame for feature engineering
X = pd.DataFrame(X, columns=california_housing.feature_names)

# Print the column names of X
print(X.columns)

# Feature Engineering
# Transform 'AveOccup' (average occupancy) if it's meant to represent something similar to 'Population'
X['AveOccup_Log'] = np.log1p(X['AveOccup'])

# Create interaction features based on  exploratory analysis
X['Rooms_per_Occupant'] = X['AveRooms'] / X['AveOccup']
X['Bedrooms_per_Room'] = X['AveBedrms'] / X['AveRooms']

# Handle outliers by capping
features_to_clip = ['AveRooms', 'AveBedrms', 'Rooms_per_Occupant', 'Bedrooms_per_Room']
for feature in features_to_clip:
    lower_bound = X[feature].quantile(0.01)
    upper_bound = X[feature].quantile(0.99)
    X[feature] = np.clip(X[feature], lower_bound, upper_bound)

# Now we also need to cap the transformed 'AveOccup_Log' feature
aveoccup_log_lower_bound = X['AveOccup_Log'].quantile(0.01)
aveoccup_log_upper_bound = X['AveOccup_Log'].quantile(0.99)
X['AveOccup_Log'] = np.clip(X['AveOccup_Log'], aveoccup_log_lower_bound, aveoccup_log_upper_bound)

# Drop the original 'AveOccup' column as we now use 'AveOccup_Log'
X.drop('AveOccup', axis=1, inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing and training the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_mse = mean_squared_error(y_test, dt_pred)

# Initializing and training the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, rf_pred)

"""**Gradient Boosting Model Befor Optimization of Parameters:**"""

# Initializing and training the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_mse = mean_squared_error(y_test, gb_pred)

"""**Gradient Boosting After Optimization of parameters:**"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

# Define the parameter distribution to sample from
param_dist = {
    'n_estimators': sp_randint(100, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': sp_randint(3, 7),
    'min_samples_split': sp_randint(2, 10),
    'min_samples_leaf': sp_randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'subsample': uniform(0.6, 0.4)
}

# Create a base model
gb = GradientBoostingRegressor(random_state=42)

# Instantiate the random search model
random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist,
                                   n_iter=100, cv=3, verbose=2, random_state=42,
                                   n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the random search to the data
random_search.fit(X_train_scaled, y_train)

# Best parameters
best_params = random_search.best_params_
print("Best parameters found: ", best_params)

# Train the best model
best_gb = GradientBoostingRegressor(**best_params, random_state=42)
best_gb.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_gb.predict(X_test_scaled)

# Evaluate the model
gb_optimized_mse = mean_squared_error(y_test, y_pred)
print("Optimized Gradient Boosting MSE: ", mse)

# Output the Mean Squared Error for each model
print("Decision Tree MSE:", dt_mse)
print("Random Forest MSE:", rf_mse)
print("Gradient Boosting MSE:", gb_mse)
print("Gradient Boosting MSE:", gb_optimized_mse)

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Evaluate additional metrics for each model
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)

    # Cross-validation scores
    cv_score = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(-cv_score)

    print(f"{name} Performance:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
    print(f"Cross-validated RMSE: {cv_score.mean():.4f} ± {cv_score.std():.4f}")

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.title(f"Residual Plot for {name}")
    plt.xlabel('Observed')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

# Evaluate Decision Tree
evaluate_model(dt_model, X_train_scaled, y_train, X_test_scaled, y_test, "Decision Tree")

# Evaluate Random Forest
evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest")

# Evaluate Gradient Boosting
evaluate_model(gb_model, X_train_scaled, y_train, X_test_scaled, y_test, "Gradient Boosting")

#Evaluate Optimized Gradient Boosting
evaluate_model(best_gb, X_train_scaled, y_train, X_test_scaled, y_test, "Optimized Gradient Boosting")