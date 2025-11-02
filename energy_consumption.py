# 온도 데이터를 이용하여 에너지 사용량 예측
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

from time import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

energy_data = pd.read_csv("COMED_hourly.csv")
energy_data["Datetime"] = pd.to_datetime(energy_data["Datetime"])
# 온도 데이터에 범위 맞추기
energy_data = energy_data[
    (energy_data["Datetime"] >= pd.to_datetime("2012-10-01"))
    & (energy_data["Datetime"] <= pd.to_datetime("2017-11-30"))
]
energy_data = energy_data.resample("D", on="Datetime").mean()
y = np.array(energy_data["COMED_MW"])

# 온도 단위: 켈빈(K)
temp_data = pd.read_csv("temperature.csv")
temp_data["datetime"] = pd.to_datetime(temp_data["datetime"])
temp_data = temp_data.resample("D", on="datetime").mean()
x = np.array(temp_data["Chicago"])
# 섭씨(°C)로 변환
x -= 273.15
x = x.reshape(-1, 1)

# 전통적인 머신러닝
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

model_1 = LinearRegression()

start_time = time()
model_1.fit(x_poly, y)
end_time = time()
learning_time_1 = end_time - start_time

y_pred_1 = model_1.predict(x_poly)

# 딥러닝
model_2 = Sequential()
model_2.add(Input(shape=(1,)))
model_2.add(Dense(32, activation="relu"))
model_2.add(Dense(16, activation="relu"))
model_2.add(Dense(1))
model_2.compile(loss="mse", optimizer="adam", metrics=["mse"])

# 10번 동안 손실에 큰 변화가 없으면 학습 중단
early_stop = EarlyStopping(
    monitor="loss", patience=10, verbose=1, restore_best_weights=True
)

start_time = time()
model_2.fit(x, y, epochs=400, batch_size=5, callbacks=[early_stop], verbose=1)
end_time = time()
learning_time_2 = end_time - start_time

y_pred_2 = model_2.predict(x, verbose=0)

print("전통적인 머신러닝 학습 시간: %ds" % learning_time_1)
print("MSE: ", mean_squared_error(y, y_pred_1))
print("MAE: ", mean_absolute_error(y, y_pred_1))
print("R2: ", r2_score(y, y_pred_1))

print("딥러닝 학습 시간: %ds" % learning_time_2)
print("MSE: ", mean_squared_error(y, y_pred_2))
print("MAE: ", mean_absolute_error(y, y_pred_2))
print("R2: ", r2_score(y, y_pred_2))

plt.plot(pd.date_range("2012-10-01", "2017-11-30", freq="D"), y, label="실제 데이터")
plt.plot(
    pd.date_range("2012-10-01", "2017-11-30", freq="D"),
    y_pred_1,
    label="머신러닝 예측 데이터",
)
plt.plot(
    pd.date_range("2012-10-01", "2017-11-30", freq="D"),
    y_pred_2,
    label="딥러닝 예측 데이터",
)
plt.legend()
plt.show()
