import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("ecommerce_data.csv")

y = df['Probability for the product to be recommended to the person']
X = df.drop('Probability for the product to be recommended to the person', axis=1)

# convert text to numbers
X = pd.get_dummies(X)

# save feature names
joblib.dump(X.columns.tolist(), "features.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model and features saved!")