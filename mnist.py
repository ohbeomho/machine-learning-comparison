import numpy as np

import struct
from array import array

from time import time

from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score


# https://www.kaggle.com/code/hojjatk/read-mnist-dataset
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return np.array(images), np.array(labels)

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


train_image_path = "train-images.idx3-ubyte"
train_label_path = "train-labels.idx1-ubyte"
test_image_path = "t10k-images.idx3-ubyte"
test_label_path = "t10k-labels.idx1-ubyte"

dataloader = MnistDataloader(
    train_image_path, train_label_path, test_image_path, test_label_path
)
(x_train, y_train), (x_test, y_test) = dataloader.load_data()

# 전통적인 머신러닝
# 2차원 배열로 변환
x_train_flatten = x_train.reshape(-1, 28 * 28)
x_test_flatten = x_test.reshape(-1, 28 * 28)

model_1 = LogisticRegression()

start_time = time()
model_1.fit(x_train_flatten, y_train)
end_time = time()
learning_time_1 = end_time - start_time

y_pred = model_1.predict(x_test_flatten)

# 딥러닝
# One-Hot 인코딩
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

model_2 = Sequential()
model_2.add(Input(shape=(28, 28, 1)))
model_2.add(Conv2D(32, (3, 3), activation="relu"))
model_2.add(MaxPooling2D((2, 2)))
model_2.add(Conv2D(64, (3, 3), activation="relu"))
model_2.add(MaxPooling2D((2, 2)))
model_2.add(Conv2D(64, (3, 3), activation="relu"))
model_2.add(Flatten())
model_2.add(Dense(64, activation="relu"))
model_2.add(Dense(10, activation="softmax"))

model_2.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

start_time = time()
model_2.fit(x_train, y_train_encoded, epochs=5, batch_size=50, verbose=1)
end_time = time()
learning_time_2 = end_time - start_time

# 모델 평가
print("전통적인 머신러닝 모델 학습 시간: %ds" % learning_time_1)
print("딥러닝 모델 학습 시간: %ds" % learning_time_2, "\n")
print("전통적인 머신러닝 모델 정확도: %.4f%%" % (accuracy_score(y_test, y_pred) * 100))
print(
    "딥러닝 모델 정확도: %.4f%%"
    % (model_2.evaluate(x_test, y_test_encoded, verbose=0)[1] * 100)
)
