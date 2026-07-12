import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("historical_data.csv")

data = data.dropna()

X = data[["Year"]]
y = data["Value"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
print("Mean Absolute Error:", mae)

future = pd.DataFrame({
    "Year": [2025, 2026, 2027, 2028, 2029]
})

future_prediction = model.predict(future)

print("\nFuture Predictions")

for year, value in zip(future["Year"], future_prediction):
    print(year, round(value, 2))

plt.scatter(X, y)
plt.plot(X, predictions)

plt.title("Predictive Analytics Using Historical Data")
plt.xlabel("Year")
plt.ylabel("Value")

plt.show()