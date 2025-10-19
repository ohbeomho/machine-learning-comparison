import numpy as np

import struct
from array import array
from os.path import join, isfile

import keras

from sklearn import svm


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


input_path = "./mnist"
train_image_path = join(input_path, "train-images.idx3-ubyte")
train_label_path = join(input_path, "train-labels.idx1-ubyte")
test_image_path = join(input_path, "t10k-images.idx3-ubyte")
test_label_path = join(input_path, "t10k-labels.idx1-ubyte")

dataloader = MnistDataloader(
    train_image_path, train_label_path, test_image_path, test_label_path
)
(x_train, y_train), (x_test, y_test) = dataloader.load_data()

# model_1 -> SVM

useFile = False

if isfile("mnist_model.keras"):
    useFile = input("Use saved model? (y/n)") == "y"

model_2 = None

if useFile:
    model_2 = keras.models.load_model("mnist_model.keras")
else:
    model_2 = keras.Sequential()
    model_2.add(
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model_2.add(keras.layers.MaxPooling2D((2, 2)))
    model_2.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model_2.add(keras.layers.MaxPooling2D((2, 2)))
    model_2.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model_2.add(keras.layers.Flatten())
    model_2.add(keras.layers.Dense(64, activation="relu"))
    model_2.add(keras.layers.Dense(10, activation="softmax"))

model_2.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model_2.fit(x_train, y_train, epochs=5, batch_size=32)
model_2.evaluate(x_test, y_test)
model_2.save("mnist_model.keras")
