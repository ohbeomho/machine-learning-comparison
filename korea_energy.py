import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

model = LinearRegression()
model.fit(
    np.array(list(map(int, train_cols))).reshape(-1, 1),
    energy_data[train_cols].tolist(),
)

energy_predict = model.predict(np.array(list(map(int, target_cols))).reshape(-1, 1))

print(
    "MSE: ",
    mean_squared_error(energy_data[target_cols].tolist(), energy_predict),
)
print(
    "MAE: ",
    mean_absolute_error(energy_data[target_cols].tolist(), energy_predict),
)
print("R2: ", r2_score(energy_data[target_cols].tolist(), energy_predict))

ax = plt.axes()
ax.xaxis.set_major_locator(tkr.MultipleLocator(12))
ax.xaxis.set_minor_locator(tkr.MultipleLocator(1))
ax.yaxis.set_major_locator(tkr.MultipleLocator(1000))
ax.yaxis.set_minor_locator(tkr.MultipleLocator(100))

ax.plot(train_cols, energy_data[train_cols].tolist())
ax.plot(target_cols, energy_data[target_cols].tolist())
ax.plot(target_cols, energy_predict)

ax.legend(("학습 데이터", "실제 데이터", "예측 데이터"))
ax.tick_params(axis="x", rotation=90)
ax.grid(axis="x")

plt.xlabel("Year")
plt.ylabel("Energy Usage (1000 toe)")

plt.show()
