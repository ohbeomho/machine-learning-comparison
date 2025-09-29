import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 결측치 없음
diabetes_data = pd.read_csv("./diabetes/diabetes_012_health_indicators_BRFSS2015.csv")
target_col = "Diabetes_012"  # 예측할 데이터
data_cols = [
    "HighBP",
    "HighChol",
    "BMI",
    "PhysActivity",
    "Fruits",
    "Veggies",
]  # 예측에 사용할 데이터

# 최근 5년 이내 콜레스테롤 검사를 받은 사람 데이터만 사용
diabetes_data = diabetes_data[diabetes_data["CholCheck"] >= 1]

# 필요한 데이터만 추출
x = diabetes_data[data_cols]
y = diabetes_data[target_col]

# 전통적인 머신러닝
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# random_state를 지정하여 실행할 때마다 같은 결과가 나오도록 함
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# K를 5로 설정
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
