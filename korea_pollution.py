import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 결측치 없음
pollution_data = pd.read_csv("./korea-pollution/south-korean-pollution-data.csv")
# 서울특별시 관악구 데이터만 사용
pollution_data = pollution_data[pollution_data["City"] == "Gwanak-Gu"]
# 날짜에 따라 정렬
pollution_data.sort_values("date", inplace=True, ignore_index=True)
dates = pollution_data["date"]
pm25 = pollution_data["pm25"]  # pm2.5
pm10 = pollution_data["pm10"]

plt.plot(dates, pm25)
plt.plot(dates, pm10)
plt.legend(("pm2.5", "pm10"))
plt.tick_params(axis="x", which="both", labelbottom=False, top=False, bottom=False)
plt.show()
