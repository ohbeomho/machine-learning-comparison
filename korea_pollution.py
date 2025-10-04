import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import koreanize_matplotlib

# 결측치 없음
pollution_data = pd.read_csv('./korea-pollution/Measurement_summary.csv')
# 서울 종로구 데이터만 사용
pollution_data = pollution_data[pollution_data['Station code'] == 101]
# 1시 데이터만 사용
pollution_data = pollution_data[pollution_data['Measurement date'].str[-5:-3] == '01']

# 부적절한 데이터 삭제 (PM2.5가 PM10보다 큰 데이터)
pollution_data = pollution_data[pollution_data['PM2.5'] < pollution_data['PM10']]


# 이상치 제거
def remove_outlier(column):
    global pollution_data
    Q1 = pollution_data[column].quantile(0.25)
    Q3 = pollution_data[column].quantile(0.75)
    IQR = Q3 - Q1

    pollution_data = pollution_data[
        (pollution_data[column] < Q3 + 1.5 * IQR)
        & (pollution_data[column] > Q1 - 1.5 * IQR)
    ]


remove_outlier('PM2.5')
remove_outlier('PM10')

# 날짜에 따라 정렬
pollution_data.sort_values('Measurement date', inplace=True, ignore_index=True)
dates = pollution_data['Measurement date']
pm25 = pollution_data['PM2.5']
pm10 = pollution_data['PM10']

ax = plt.axes()

ax.plot(dates, pm10)
ax.plot(dates, pm25)
ax.legend(('PM2.5', 'PM10'))
ax.tick_params(axis='x', which='minor', labelbottom=False, top=False, bottom=False)
ax.tick_params(axis='x', rotation=90)
ax.xaxis.set_major_locator(MultipleLocator(10))

plt.xlabel('날짜')
plt.ylabel('PM2.5/PM10 (μg/m3)')
plt.show()
