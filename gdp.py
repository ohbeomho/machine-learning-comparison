import pandas as pd

gdp_data = pd.read_csv("./gdp/gdp.csv")
# 결측치 제거
gdp_data.dropna(axis=0, inplace=True)
