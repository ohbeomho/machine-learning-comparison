import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import koreanize_matplotlib
from time import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import keras

# 결측치 없음
# 총 에너지 사용량만 사용
energy_data = pd.read_csv("./korea-energy/EnergyUsage_bySector.csv").loc[0]
energy_data = energy_data[1:].apply(
    lambda x: float(x.replace(",", ""))
)  # 문자열로 저장된 데이터를 숫자로 변환

# 학습에 사용할 데이터 (1997년 1월 ~ 2014년 12월)
train_cols = []

for i in range(1997, 2015):
    for j in range(1, 13):
        train_cols.append("%d%02d" % (i, j))

# 예측에 사용할 데이터 (2015년 1월 ~ 2021년 11월)
target_cols = []

for i in range(2015, 2022):
    for j in range(1, 13):
        target_cols.append("%d%02d" % (i, j))

target_cols.pop()

# 2000년 1월 -> 2000 + 1/12
x_train = np.array(
    list(map(float, map(lambda x: float(x[:4]) + float(x[4:]) / 12, train_cols)))
).reshape(-1, 1)
y_train = np.array(energy_data[train_cols])
x_test = np.array(
    list(map(float, map(lambda x: float(x[:4]) + float(x[4:]) / 12, target_cols)))
).reshape(-1, 1)
y_test = np.array(energy_data[target_cols])

# 데이터 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# --- 전통적인 머신러닝 ---
print("전통적인 머신러닝 학습 시작")
model_1 = LinearRegression()

start_time = time()
model_1.fit(x_train, y_train)
end_time = time()
learning_time_1 = end_time - start_time
y_pred_1 = model_1.predict(x_test)

# --- 딥러닝 ---

print("딥러닝 학습 시작")
model_2 = keras.Sequential()
model_2.add(keras.layers.Dense(32, activation="relu", input_shape=(1,)))
model_2.add(keras.layers.Dense(32, activation="relu"))
model_2.add(keras.layers.Dense(1))
model_2.compile(loss="mse", optimizer="adam", metrics=["mse"])

start_time = time()
model_2.fit(x_train, y_train, epochs=250, batch_size=2)
end_time = time()
learning_time_2 = end_time - start_time

y_pred_2 = model_2.predict(x_test)
model_2.save("korea_energy_model.keras")

print("전통적인 머신러닝 학습 시간: %ds" % learning_time_1)
print("딥러닝 모델 학습 시간: %ds" % learning_time_2)

print("전통적인 머신러닝 모델 평가")
print(
    "MSE: ",
    mean_squared_error(y_test, y_pred_1),
)
print(
    "MAE: ",
    mean_absolute_error(y_test, y_pred_1),
)
print("R2: ", r2_score(y_test, y_pred_1), "\n")

print("딥러닝 모델 평가")
print("MSE: ", mean_squared_error(y_test, y_pred_2))
print("MAE: ", mean_absolute_error(y_test, y_pred_2))
print("R2: ", r2_score(y_test, y_pred_2))

ax = plt.axes()
ax.xaxis.set_major_locator(tkr.MultipleLocator(12))
ax.xaxis.set_minor_locator(tkr.MultipleLocator(1))
ax.yaxis.set_major_locator(tkr.MultipleLocator(1000))
ax.yaxis.set_minor_locator(tkr.MultipleLocator(100))

ax.plot(train_cols, energy_data[train_cols].tolist())
ax.plot(target_cols, y_test)
ax.plot(target_cols, y_pred_1)
ax.plot(target_cols, y_pred_2)

ax.legend(("학습 데이터", "실제 데이터", "머신러닝 예측 데이터", "딥러닝 예측 데이터"))
ax.tick_params(axis="x", rotation=90)
ax.grid(axis="x")

plt.xlabel("날짜")
plt.ylabel("에너지 사용량 (1000 toe)")

plt.show()
