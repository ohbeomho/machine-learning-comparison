import pandas as pd

gdp_data = pd.read_csv('./gdp/gdp.csv')
# 결측치 제거
gdp_data.dropna(axis=0, inplace=True)
# 학습에 사용할 데이터 (1960~2010년)
train_col = range(1960, 2011)
# 에측할 데이터 (2011~2020년)
target_col = range(2011, 2021)