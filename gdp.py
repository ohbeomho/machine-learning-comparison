import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras

gdp_data = pd.read_csv("./gdp/gdp.csv")

# 학습할 국가
target_country = "Korea, Rep."
train_data = gdp_data[gdp_data["Country Name"] == target_country]

# 학습에 사용할 데이터 (1960~2010년)
train_cols = list(map(str, range(1960, 2011)))
# 예측할 데이터 (2011~2020년)
target_cols = list(map(str, range(2011, 2021)))

train_data = train_data[train_cols + target_cols]
train_data /= 1e8  # 100000000 (1억) 으로 나눠서 숫자 작게 하기

x_train = np.array(list(map(int, train_cols))).reshape(-1, 1)
y_train = train_data[train_cols].values.reshape(-1, 1)
x_test = np.array(list(map(int, target_cols))).reshape(-1, 1)
y_test = train_data[target_cols].values.reshape(-1, 1)

# 2차식으로 선형회귀
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

model_1 = LinearRegression()
model_1.fit(x_train_poly, y_train)

y_pred_1 = model_1.predict(x_test_poly)
pickle.dump(model_1, open("gdp_model.pkl", "wb"))

# 모델 평가
print("전통적인 머신러닝 평가")
print("MSE: ", mean_squared_error(y_test.flatten(), y_pred_1))
print(
    "MAE: ",
    mean_absolute_error(y_test.flatten(), y_pred_1),
)
print("R2: ", r2_score(y_test.flatten(), y_pred_1), "\n")

model_2 = keras.Sequential()
model_2.add(keras.layers.Dense(64, activation="relu", input_shape=(1,)))
model_2.add(keras.layers.Dense(64, activation="relu"))
model_2.add(keras.layers.Dense(1))

model_2.compile(loss="mse", optimizer="adam", metrics=["mse"])

model_2.fit(x_train, y_train, epochs=5)

y_pred_2 = model_2.predict(x_test)
model_2.save("gdp_model.keras")

print("딥러닝 모델 평가")
print("MSE: ", mean_squared_error(y_test.flatten(), y_pred_2))
print(
    "MAE: ",
    mean_absolute_error(y_test.flatten(), y_pred_2),
)
print("R2: ", r2_score(y_test.flatten(), y_pred_2))

# 그래프로 결과 시각화
plt.plot(train_cols, train_data[train_cols].values.flatten())
plt.plot(target_cols, y_test.flatten())
plt.plot(target_cols, y_pred_1)
plt.plot(target_cols, y_pred_2)
plt.legend(("학습 데이터", "실제 데이터", "머신러닝 예측 데이터", "딥러닝 예측 데이터"))
plt.xticks(rotation=90)
plt.xlabel("년도")
plt.ylabel("GDP (억 $)")
plt.show()
