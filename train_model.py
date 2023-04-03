import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

import config
from dataset_preparation import get_dataset


def create_model(input_shape: tuple):
    """
    create architecture and compile model
    :param input_shape: input shape in model
    :return: compiled model
    """
    In = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding="same", activation="relu")(In)
    x = Conv2D(32, (5, 5), activation="relu")(x)
    x = Conv2D(32, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = Conv2D(64, (5, 5), activation="relu")(x)
    x = Conv2D(64, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    Out = Dense(28, activation="softmax")(x)
    model = Model(In, Out)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_model(model, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, plot_model_arch: bool,
                path_log: str = config.path_log, path_model: str = config.path_model) -> None:
    """
    function for train model
    :param model: compiled model
    :param x_train: train data dataset
    :param y_train: train labels dataset
    :param x_test: test data dataset
    :param y_test: test labels dataset
    :param plot_model_arch: plot the architecture of model
    :param path_log: path to save train log
    :param path_model: path to save model
    :return: None
    """
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    train_gen = datagen.flow(x_train, y_train, batch_size=config.batch_size)
    test_gen = datagen.flow(x_test, y_test, batch_size=config.batch_size)

    if plot_model_arch:
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    model_checkpoint_callback = ModelCheckpoint(filepath=path_model,
                                                monitor='val_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max')
    csv_logger = CSVLogger(path_log)
    model.fit(train_gen,
              epochs=config.epochs,
              verbose=1,
              steps_per_epoch=x_train.shape[0] // config.batch_size,
              validation_data=test_gen,
              validation_steps=x_test.shape[0] // config.batch_size,
              callbacks=[model_checkpoint_callback, csv_logger])
    return None


def plot_history_train(path_log: str) -> None:
    """
    plot training log with metrics
    :param path_log: path to train log
    :return:
    """
    df = pd.read_csv(path_log, sep=',', engine='python')
    print(df)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(df["accuracy"])
    plt.plot(df["val_accuracy"])
    plt.legend(["accuracy", "val_accuracy"])
    plt.subplot(1, 2, 2)
    plt.plot(df["loss"])
    plt.plot(df["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.show()
    return None
