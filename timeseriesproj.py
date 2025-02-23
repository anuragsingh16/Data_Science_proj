# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, confusion_matrix, 
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Time Series Analysis (Sunspots Dataset)
# --------------------------

# Load dataset
from statsmodels.datasets import sunspots
data = sunspots.load_pandas().data

# Convert year to datetime format
data['YEAR'] = data['YEAR'].astype(int)
data['Date'] = pd.to_datetime(data['YEAR'].astype(str), format='%Y')
data.set_index('Date', inplace=True)
data.drop('YEAR', axis=1, inplace=True)
data.columns = ['Sunspots']

# EDA Plot
plt.figure(figsize=(12,6))
plt.plot(data)
plt.title('Sunspot Numbers Over Time (1700-2008)')
plt.xlabel('Year')
plt.ylabel('Number of Sunspots')
plt.show()

# Stationarity Check
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    
check_stationarity(data['Sunspots'])

# Decomposition
decomposition = seasonal_decompose(data['Sunspots'], period=11)  # 11-year solar cycle
decomposition.plot()
plt.show()

# Train-test split
train = data[:'1990']
test = data['1991':]

# ARIMA Model
model = ARIMA(train, order=(2,1,2))
results = model.fit()
print(results.summary())

# Forecasting
forecast_steps = len(test)
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# Evaluation
rmse = np.sqrt(np.mean((forecast_mean - test['Sunspots'])**2))
print(f'RMSE: {rmse}')

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(train, label='Training')
plt.plot(test, label='Actual')
plt.plot(forecast_mean, label='Forecast')
plt.title('Sunspots Forecast with ARIMA')
plt.legend()
plt.show()

# --------------------------
# Classification (Wine Dataset)
# --------------------------

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# EDA
print(X.describe())

plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)

# XGBoost Classifier
xgb = XGBClassifier(random_state=42)
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}

grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5)
grid_xgb.fit(X_train_scaled, y_train)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f'\n{model_name} Evaluation:')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred, average='weighted'))
    print('Recall:', recall_score(y_true, y_pred, average='weighted'))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=wine.target_names)
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_xgb, 'XGBoost')

# Feature Importance
plt.figure(figsize=(10,6))
sorted_idx = best_rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], best_rf.feature_importances_[sorted_idx])
plt.title('Random Forest Feature Importance')
plt.show()

# --------------------------
# Model Saving
# --------------------------

with open('arima_model.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

print("Models saved successfully")

# --------------------------
# Deployment Example (Flask API snippet)
# --------------------------
"""
# Example Flask API endpoint
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = rf_model.predict(features)
    return jsonify({'class': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
"""
