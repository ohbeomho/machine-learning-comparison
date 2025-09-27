# 이미지 파일이 많아 github 에 올릴 수 없음
# 이미지 파일들을 npy 파일로 변환하는 코드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from os import listdir

input_path = join("cifar10", "cifar10_images")
target_path = "cifar10"

# 사진 종류
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_images():
    images = {"train": [], "test": []}
    labels = {"train": [], "test": []}

    for type in ["train", "test"]:
        for cls in classes:
            images_path = join(input_path, type, cls)
            filenames = listdir(images_path)

            for filename in filenames:
                img = plt.imread(join(images_path, filename))
                images[type].append(img)

            labels[type].extend([cls] * len(filenames))

    return ((images["train"], images["test"]), (labels["train"], labels["test"]))


((train_images, test_images), (train_labels, test_labels)) = read_images()

np.save(join(target_path, "train_images.npy"), np.array(train_images))
np.save(join(target_path, "test_images.npy"), np.array(test_images))
pd.DataFrame(train_labels).to_csv(join(target_path, "train_labels.csv"))
pd.DataFrame(test_labels).to_csv(join(target_path, "test_labels.csv"))
