import numpy as np
import tensorflow as tf
import cv2
from imutils import build_montages
import matplotlib.pyplot as plt

import config
from dataset_preparation import get_dataset


def predict_test_dataset():
    x_train, y_train, x_test, y_test = get_dataset()
    model = tf.keras.models.load_model(config.path_model)
    names_labels = config.arabic_characters
    images = []

    for i in np.random.choice(np.arange(0, len(y_test)), size=(49,)):
        probs = model.predict(x_test[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = names_labels[prediction[0]]
        label_true = names_labels[y_test[i].argmax(axis=0)]
        image = (x_test[i] * 255).astype("uint8")
        color = (0, 255, 0)
        if prediction[0] != y_test[i].argmax(axis=0):
            color = (255, 0, 0)
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (110, 110), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(image, label_true, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        images.append(image)
        montage = build_montages(images, (110, 110), (7, 7))[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(montage)
    plt.show()
    plt.savefig(config.path_predict_plot)


predict_test_dataset()
