import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, x_test) = (
    np.load("./cifar10/train_images.npy"),
    np.load("./cifar10/test_images.npy"),
)
(y_train, y_test) = (
    pd.read_csv("./cifar10/train_labels.csv"),
    pd.read_csv("./cifar10/test_labels.csv"),
)
