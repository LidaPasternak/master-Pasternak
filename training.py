import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('ship_fuel_efficiency.csv')
df = df.drop('ship_id', axis=1)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
df[numeric_cols] = np.log(df[numeric_cols])
df = pd.get_dummies(df, columns=categorical_cols)

columns_to_keep = [
    'fuel_type_Diesel',
    'ship_type_Oil Service Boat',
    'ship_type_Tanker Ship',
    'ship_type_Surfer Boat',
    'distance',
    'fuel_consumption'
]
final_columns = [col for col in columns_to_keep if col in df.columns]
df_final = df[final_columns]

X = df_final.drop(columns=['fuel_consumption'])
y = df_final['fuel_consumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[['distance']] = scaler.fit_transform(X_train[['distance']])

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

train_preds_scaled = model.predict(X_train_scaled)
train_preds_log = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
residuals_log = y_train.values - train_preds_log
smearing_factor = np.mean(np.exp(residuals_log))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('y_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
with open('smearing_factor.pkl', 'wb') as f:
    pickle.dump(smearing_factor, f)
