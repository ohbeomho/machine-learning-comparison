import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import koreanize_matplotlib

# 결측치 없음
# 총 에너지 사용량만 사용
energy_data = pd.read_csv("./korea-energy/EnergyUsage_bySector.csv").loc[0]
energy_data = energy_data[1:].apply(lambda x: float(x.replace(",", "")))

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


ax = plt.axes()
ax.xaxis.set_major_locator(tkr.MultipleLocator(12))
ax.xaxis.set_minor_locator(tkr.MultipleLocator(1))
ax.yaxis.set_major_locator(tkr.MultipleLocator(1000))
ax.yaxis.set_minor_locator(tkr.MultipleLocator(100))

ax.plot(train_cols, energy_data[train_cols].tolist())
ax.plot(target_cols, energy_data[target_cols].tolist())

ax.legend(("학습 데이터", "실제 데이터"))
ax.tick_params(axis="x", rotation=90)
ax.grid(axis="x")

plt.xlabel("Year")
plt.ylabel("Energy Usage (1000 toe)")

plt.show()
