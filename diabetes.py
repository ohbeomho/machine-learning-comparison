import pandas as pd

# 결측치 없음
diabetes_data = pd.read_csv('./diabetes/diabetes_012_health_indicators_BRFSS2015.csv')
target_col = 'Diabetes_012' # 예측할 데이터
