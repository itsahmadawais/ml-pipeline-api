import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")

# DATA
#  → X, y split
#  → train/test split
#  → choose model
#  → fit (learn)
#  → predict (guess)
#  → evaluate (check error)

# print(df.head())

X = df[["house_size"]]  # 2D table
y = df["house_price"] # 1D list


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict

y_pred = model.predict(X_test)

print("Predicted values:", y_pred)
print("Actual values:", y_test.values)

# Evaluate
error = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", error)

plt.scatter(X,y)
plt.plot(X, model.predict(X))
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.show()