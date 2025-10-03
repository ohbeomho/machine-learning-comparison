import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


# 2차식으로 선형회귀
poly = PolynomialFeatures(degree=2)
train_poly = poly.fit_transform(np.array(list(map(int, train_cols))).reshape(-1, 1))
test_poly = poly.transform(np.array(list(map(int, target_cols))).reshape(-1, 1))

model = LinearRegression()
model.fit(
    train_poly,
    train_data[train_cols].values.reshape(-1, 1),
)
gdp_predict = model.predict(test_poly)

# 그래프로 결과 시각화

# 모델 평가
print(
    "MSE: ", mean_squared_error(train_data[target_cols].values.flatten(), gdp_predict)
)
print(
    "MAE: ", mean_absolute_error(train_data[target_cols].values.flatten(), gdp_predict)
)
print("R2: ", r2_score(train_data[target_cols].values.flatten(), gdp_predict))

plt.plot(train_cols, train_data[train_cols].values.flatten())
plt.plot(target_cols, train_data[target_cols].values.flatten())
plt.plot(target_cols, gdp_predict)
plt.legend(("학습 데이터", "실제 데이터", "예측 데이터"))
plt.xticks(rotation=90)
plt.xlabel("년도")
plt.ylabel("GDP (억 $)")
plt.show()
