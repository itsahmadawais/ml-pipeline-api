import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

def load_data(path):
    return pd.read_csv(path)


def train_model(df):
    X = df[["house_size", "bedrooms", "car_space"]]
    y = df["house_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)

    # Save Model
    joblib.dump(model, "house_model.pkl")

    return model, error


def predict_house_price(model, size, bedrooms, car_space):
    new_house = pd.DataFrame(
        [[size, bedrooms, car_space]],
        columns=["house_size", "bedrooms", "car_space"]
    )

    return model.predict(new_house)[0]


df = load_data("../data/house_data.csv")

model, error = train_model(df)
print("MSE:", error)

price = predict_house_price(model, 1700, 5, 2)
print("Predicted:", price)