from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from dataset_preparation import get_dataset


def create_model(input_shape: tuple):
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


x_train, y_train, x_test, y_test = get_dataset()
print(x_train.shape)
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

print(train_gen)
model = create_model((32, 32, 1))
model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model_checkpoint_callback = ModelCheckpoint(filepath='model.h5',
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
csv_logger = CSVLogger('model_training.log')

model.fit(train_gen,
          epochs=config.epochs,
          verbose=1,
          steps_per_epoch=x_train.shape[0] // config.batch_size,
          validation_data=test_gen,
          validation_steps=x_test.shape[0] // config.batch_size,
          callbacks=[model_checkpoint_callback, csv_logger])
