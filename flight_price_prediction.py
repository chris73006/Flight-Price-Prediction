import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("Clean_Dataset.csv")

df.drop("Unnamed: 0", axis=1, inplace=True) #drops unwanted column,Unnamed

X = df.drop("price", axis=1)
y = df["price"]

le = LabelEncoder() #converts text to numbers

for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

#Model Training

model.fit(X_train, y_train)

print("Model trained successfully")
y_pred = model.predict(X_test)

print(y_pred[:5])

#Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

#Actual vs Predicted Flight Prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Flight Prices")

plt.show()

#Feature Importance Analysis
importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.show()