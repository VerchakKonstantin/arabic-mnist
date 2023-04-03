import pandas as pd
from tensorflow.keras.utils import to_categorical
import config


def get_dataset(path_train_x: str = config.path_train_x, path_train_y: str = config.path_train_y,
                path_test_x: str = config.path_test_x, path_test_y: str = config.path_test_y,
                arabic_characters: list = config.arabic_characters):
    """
    function for load and preparation dataset
    :param path_train_x: path to train csv images
    :param path_train_y: path to train csv labels
    :param path_test_x: path to test csv images
    :param path_test_y: path to test scv labels
    :param arabic_characters: decryption of labels
    :return: arrays of dataset
    """

    x_train = pd.read_csv(path_train_x, header=None).to_numpy()
    y_train = pd.read_csv(path_train_y, header=None).to_numpy() - 1

    x_test = pd.read_csv(path_test_x, header=None).to_numpy()
    y_test = pd.read_csv(path_test_y, header=None).to_numpy() - 1

    x_train = x_train.reshape(-1, 32, 32, 1)
    x_test = x_test.reshape(-1, 32, 32, 1)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test

