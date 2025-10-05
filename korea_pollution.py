import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import koreanize_matplotlib

# 결측치 없음
pollution_data = pd.read_csv('./korea-pollution/Measurement_summary.csv')
# 서울 종로구 데이터만 사용
pollution_data = pollution_data[pollution_data['Station code'] == 101]
# 오후 1시 데이터만 사용
pollution_data = pollution_data[pollution_data['Measurement date'].str[-5:-3] == '13']


# 이상치 제거
def remove_outliers(column):
    global pollution_data
    Q1 = pollution_data[column].quantile(0.25)
    Q3 = pollution_data[column].quantile(0.75)
    IQR = Q3 - Q1

    pollution_data = pollution_data[
        (pollution_data[column] >= Q1 - 2 * IQR)
        & (pollution_data[column] <= Q3 + 2 * IQR)
    ]


remove_outliers('PM2.5')

# 날짜에 따라 정렬
pollution_data.sort_values('Measurement date', inplace=True, ignore_index=True)
dates = pollution_data['Measurement date']
pm25 = pollution_data['PM2.5']

test_size = int(0.2 * len(dates))
train_size = len(dates) - test_size

poly = PolynomialFeatures(degree=3)
train_poly = poly.fit_transform(np.arange(train_size).reshape(-1, 1))
test_poly = poly.transform(np.arange(train_size, train_size + test_size).reshape(-1, 1))

model = LinearRegression()
model.fit(
    train_poly,
    np.array(pm25[:train_size]).reshape(-1, 1),
)
pm25_pred = model.predict(test_poly)

ax = plt.axes()

ax.plot(dates[:train_size], pm25[:train_size])
ax.plot(dates[train_size:], pm25[train_size:])
ax.plot(dates[train_size:], pm25_pred)

ax.legend(
    (
        '학습 데이터',
        '실제 값',
        '예측 데이터',
    )
)

ax.tick_params(axis='x', which='minor', labelbottom=False, top=False, bottom=False)
ax.tick_params(axis='x', rotation=90)
ax.xaxis.set_major_locator(MultipleLocator(10))

plt.xlabel('날짜')
plt.ylabel('PM2.5 (μg/m3)')
plt.show()
