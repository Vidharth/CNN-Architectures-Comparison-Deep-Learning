"""Deep Learning Project 1"""

import gc
import os
import time

import numpy as np
import pandas as pd
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(trainpath, testpath, imageheight, imagewidth, batchsize, valsplit):
    """
    Method to load the dataset.
    :param trainpath: path to the training data.
    :param testpath: path to the testing data.
    :param imageheight: height of the image.
    :param imagewidth: width of the image.
    :param batchsize: number of batches for the images.
    :param valsplit: amount of data to be reserved for validation.
    :return: Returns the training data, validation data, and testing data.
    """
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=valsplit)
    train_data = data_gen.flow_from_directory(trainpath, target_size=(imageheight, imagewidth), batch_size=batchsize,
                                              subset="training")
    validation_data = data_gen.flow_from_directory(trainpath, target_size=(imageheight, imagewidth),
                                                   batch_size=batchsize, subset="validation")
    test_data = data_gen.flow_from_directory(testpath, target_size=(imageheight, imagewidth), batch_size=batchsize)

    return train_data, validation_data, test_data


def model_lenet5(optimizer, activation, imageheight, imagewidth, channels, padding):
    """
    LeNet5 Architecture.
    :param optimizer: Optimizer value.
    :param activation: activation value.
    :param imageheight: height of the image.
    :param imagewidth: width of the image.
    :param channels: number of channels for the image.
    :param padding: padding value True/False
    :return: returns the built and compiled model.
    """
    model = Sequential()
    model.add(Convolution2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation=activation,
                            padding=padding, input_shape=(imageheight, imagewidth, channels)))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation=activation, padding=padding))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    model.add(Dense(84, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(6, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def model_alexnet(optimizer, activation, imageheight, imagewidth, channels, padding):
    """
    AlexNet Architecture.
    :param optimizer: Optimizer value.
    :param activation: activation value.
    :param imageheight: height of the image.
    :param imagewidth: width of the image.
    :param channels: number of channels for the image.
    :param padding: padding value True/False
    :return: returns the built and compiled model.
    """
    model = Sequential()
    model.add(Convolution2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=activation,
                            padding=padding, input_shape=(imageheight, imagewidth, channels)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation=activation, padding=padding))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Convolution2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding=padding))
    model.add(Convolution2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding=padding))
    model.add(Convolution2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(6, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def run_model(modelname, train, val, test, optimizer, activation, epochs, imageheight, imagewidth, channels,
              columns, padding):
    """
    Method to run the model.
    :param modelname: name of the model.
    :param train: train data.
    :param val: validation data.
    :param test: test data.
    :param optimizer: optimizer value.
    :param activation: activation value.
    :param epochs: number of epochs.
    :param imageheight: height of the image.
    :param imagewidth: width of the image.
    :param channels: number of channels of the image.
    :param columns: column values for the output results data.
    :param padding: padding True/False
    :return: returns the result of the model, the accuracy and loss metrics.
    """
    es = EarlyStopping(patience=10, verbose=1)

    if modelname == "LeNet5":
        model = model_lenet5(optimizer, activation, imageheight, imagewidth, channels, padding)
    elif modelname == "AlexNet":
        model = model_alexnet(optimizer, activation, imageheight, imagewidth, channels, padding)
    else:
        model = None
        print("Model Name does not Exist!")

    model.fit_generator(train, epochs=epochs, validation_data=val, verbose=0, callbacks=[es])
    train_loss, train_accuracy = model.evaluate_generator(train)
    val_loss, val_accuracy = model.evaluate_generator(val)
    test_loss, test_accuracy = model.evaluate_generator(test)
    result = {columns[0]: train_loss, columns[1]: train_accuracy, columns[2]: val_loss, columns[3]: val_accuracy,
              columns[4]: test_loss, columns[5]: test_accuracy}

    return result


if __name__ == "__main__":
    Count = 1
    ImageHeight = 150
    ImageWidth = 150
    Channels = 3
    BatchSize = 64
    ValidationSplit = 0.3
    Epochs = 100
    Padding = "same"
    ModelName = "AlexNet"
    TrainPath = "dataset//seg_train"
    TestPath = "dataset//seg_test"
    CSVPath = str(ModelName) + ".csv"

    Optimizers = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
    Activations = ["linear", "exponential", "sigmoid", "hard_sigmoid", "tanh", "relu", "softsign", "softplus", "selu",
                   "softmax", "elu"]
    CSVColumns = ["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Test Loss",
                  "Test Accuracy", "Parameters"]

    Train, Val, Test = load_data(trainpath=TrainPath, testpath=TestPath, imageheight=ImageHeight, imagewidth=ImageWidth,
                                 batchsize=BatchSize, valsplit=ValidationSplit)

    Data = pd.DataFrame(columns=CSVColumns)
    if not os.path.isfile(".//" + str(CSVPath)):
        Data.to_csv(CSVPath)

    k.clear_session()

    for Opt in Optimizers:
        for Act in Activations:
            StartTime = time.time()
            Results = run_model(modelname=ModelName, train=Train, val=Val, test=Test, optimizer=Opt, activation=Act,
                                epochs=Epochs, imageheight=ImageHeight, imagewidth=ImageWidth, channels=Channels,
                                columns=CSVColumns, padding=Padding)
            Results["Parameters"] = [str(Count) + ". " + "Optimizer: " + str(Opt) + ", " + "Activation: " + str(Act)]
            data = pd.DataFrame(Results)
            data.to_csv(CSVPath, mode="a", header=False)
            ElapsedTime = time.time() - StartTime
            print("Run: " + str(Count) + "/" + str(len(Optimizers) * len(Activations)))
            print("Run Time: " + str(np.round(ElapsedTime / 60)) + " " + "Minutes.")
            print("Time Remaining In Hours: " + str(
                np.round((ElapsedTime * ((len(Optimizers) * len(Activations)) - Count)) / (60 * 60))) + " " + "Hours.")
            print("Time Remaining In Minutes: " + str(
                np.round((ElapsedTime * ((len(Optimizers) * len(Activations)) - Count)) / 60)) + " " + "Minutes.")
            print(Results)
            Count += 1
            k.clear_session()
            gc.collect()
