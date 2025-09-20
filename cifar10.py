import pandas as pd
import numpy as np
from os.path import join

<<<<<<< HEAD
# TODO: CIFAR-10 이미지 파일 다운받기
input_path = "./cifar-10"
||||||| parent of 2241fd9 (More datasets)
=======
# Fix later
# It only loads the first image
# Maybe I should use the original data... (this is numpy compressed version)
input_path = "./cifar-10"
train_data = np.load(join(input_path, "train.npy"))
test_data = np.load(join(input_path, "test.npy"))

>>>>>>> 2241fd9 (More datasets)
train_labels = pd.read_csv(join(input_path, "trainLabels.csv"))
