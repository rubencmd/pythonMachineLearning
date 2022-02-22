from re import VERBOSE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random

data = pd.read_csv("drebin-215-dataset-5560malware-9476-benign.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, 215]

imputeValues = [0, 1]

X_impute = X.replace("?", random.choice(imputeValues))

X_pad = X_impute.assign(A=0, B=0, C=0, D=0, E=0, F=0, G=0, H=0, I=0, J=0)

array_X_pad = X_pad.to_numpy()

labels = y.to_numpy()

length_array = len(array_X_pad)

imgNum = 0

for c in range(length_array):
    k = 0
    imageStore = np.zeros((15, 15))
    testImage = array_X_pad[c]
    for i in range(15):
        for j in range(15):
            imageStore[i][j] = testImage[k] * 225
            k += 1

    im = Image.fromarray(imageStore)
    if im.mode != "RGB":
        im = im.convert("RGB")

    if labels[c] == "S":
        im.save(f"derbin_images/S_Malware/img_{imgNum:04d}.png")
    else:
        im.save(f"derbin_images/B_Benign/img_{imgNum:04d}.png")

    imgNum += 1


batch_size = 32
img_height = 15
img_width = 15

dataset_train = tf.keras.utils.image_dataset_from_directory(
    "derbin_images/",
    image_size=(img_height, img_width),
    shuffle=True,
    seed=12,
    validation_split=0.1,
    subset="training",
)

dataset_validation = tf.keras.utils.image_dataset_from_directory(
    "derbin_images/",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=12,
    validation_split=0.1,
    subset="validation",
)

dataset_testing = tf.keras.utils.image_dataset_from_directory(
    "derbin_images/",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=12,
    validation_split=0.05,
    subset="validation",
)

class_names = dataset_train.class_names

fig = plt.figure(figsize=(15, 15))
for images, labels in dataset_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))


plt.show()


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    ]
)

model.compute_output_shape(input_shape=(None, 15, 15, 3))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.compute_output_shape(input_shape=(None, 15, 15, 3))
model.add(tf.keras.layers.Dense(1, activation=None))
tf.random.set_seed(1)

model.build(input_shape=(None, 15, 15, 3))

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    dataset_train,
    validation_data=dataset_validation,
    epochs=20,
)

hist = history.history
x_arr = np.arange(len(hist["loss"])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist["loss"], "-o", label="Train loss")
ax.plot(x_arr, hist["val_loss"], "--<", label="Validation loss")
ax.legend(fontsize=15)
ax.set_xlabel("Epoch", size=15)
ax.set_ylabel("Loss", size=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist["accuracy"], "-o", label="Train acc.")
ax.plot(x_arr, hist["val_accuracy"], "--<", label="Validation acc.")
ax.legend(fontsize=15)
ax.set_xlabel("Epoch", size=15)
ax.set_ylabel("Accuracy", size=15)

plt.show()

hist2 = history.history
x_arr = np.arange(len(hist["loss"] + hist2["loss"]))


fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist["loss"] + hist2["loss"], "-o", label="Train Loss")
ax.plot(x_arr, hist["val_loss"] + hist2["val_loss"], "--<", label="Validation Loss")
ax.legend(fontsize=15)


ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist["accuracy"] + hist2["accuracy"], "-o", label="Train Acc.")
ax.plot(
    x_arr, hist["val_accuracy"] + hist2["val_accuracy"], "--<", label="Validation Acc."
)
ax.legend(fontsize=15)
plt.show()

results = model.evaluate(dataset_testing, verbose=0)
print("Test Acc: {:.2f}%".format(results[1] * 100))
