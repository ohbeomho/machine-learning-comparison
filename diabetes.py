import pandas as pd
from time import time

# 결측치 없음
diabetes_data = pd.read_csv(
    "./diabetes/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)
target_col = "Diabetes_binary"  # 예측할 데이터
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

# --- 전통적인 머신러닝 ---
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# random_state를 지정하여 실행할 때마다 같은 결과가 나오도록 함
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# K를 5로 설정
model_1 = KNeighborsClassifier(n_neighbors=5)

start_time = time()
model_1.fit(x_train, y_train)
end_time = time()
learning_time_1 = end_time - start_time

y_pred = model_1.predict(x_test)

# --- 딥러닝 ---
from keras.models import Sequential
from keras.layers import Dense

model_2 = Sequential()
model_2.add(Dense(64, activation="relu", input_dim=6))
model_2.add(Dense(64, activation="relu"))
model_2.add(Dense(1, activation="sigmoid"))

model_2.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "recall"]
)

start_time = time()
model_2.fit(x_train, y_train, epochs=5, verbose=1)
end_time = time()
learning_time_2 = end_time - start_time

model_2_eval = model_2.evaluate(x_test, y_test, verbose=0)

from sklearn.metrics import accuracy_score, recall_score

# 모델 평가
print("전통적인 머신러닝 학습 시간: %ds" % learning_time_1)
print("딥러닝 모델 학습 시간: %ds" % learning_time_2, "\n")
print("전통적인 머신러닝")
print("정확도: %.4f%%" % (accuracy_score(y_test, y_pred) * 100))
print(
    "재현율: %.4f%%" % (recall_score(y_test, y_pred) * 100),
)
print()
print("딥러닝")
print("정확도: %.4f%%" % (model_2_eval[1] * 100))
print(
    "재현율: %.4f%%" % (model_2_eval[2] * 100),
)
