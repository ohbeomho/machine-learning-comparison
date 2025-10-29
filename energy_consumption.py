import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

from time import time

from sklearn.model_selection import train_test_split

energy_data = pd.read_csv("./energy_consumption/COMED_hourly.csv")

X_train, X_test, y_train, y_test = train_test_split(
    energy_data["Datetime"], energy_data["COMED_MW"], test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
