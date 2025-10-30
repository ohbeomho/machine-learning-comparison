import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

from time import time

energy_data = pd.read_csv("./energy_consumption/COMED_hourly.csv")
energy_data["Datetime"] = pd.to_datetime(energy_data["Datetime"])
energy_data = energy_data.resample("D", on="Datetime").mean()

x = energy_data.index.astype(np.int64)
y = energy_data["COMED_MW"]

test_size = 0.1

x_train = x[: -int(len(x) * test_size)]
x_test = x[-int(len(x) * test_size) :]
y_train = y[: -int(len(x) * test_size)]
y_test = y[-int(len(x) * test_size) :]

(x_train, x_test, y_train, y_test) = (
    np.array(x_train).reshape(-1, 1),
    np.array(x_test).reshape(-1, 1),
    np.array(y_train).reshape(-1, 1),
    np.array(y_test).reshape(-1, 1),
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 전통적인 머신러닝
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

model_1 = LinearRegression()

start_time = time()
model_1.fit(x_train_poly, y_train)
end_time = time()
learning_time_1 = end_time - start_time

y_pred_1 = model_1.predict(x_test_poly)

print("MSE: ", mean_squared_error(y_test, y_pred_1))
print("MAE: ", mean_absolute_error(y_test, y_pred_1))
print("R2: ", r2_score(y_test, y_pred_1))

plt.plot(x_train, y_train, label="학습 데이터")
plt.plot(x_test, y_test, label="실제 데이터")
plt.plot(x_test, y_pred_1, label="머신러닝 예측 데이터")
plt.legend()
plt.show()
