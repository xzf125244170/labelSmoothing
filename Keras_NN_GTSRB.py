
from keras.datasets import mnist
from keras.optimizers import adam
from keras.layers import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import os
import csv
import pdb

import numpy as np

def main():

    # accuracyVector = []
    # scoreVector = []
    # for loop in range(1, 2):
        # score, acc = oneLayerNN(numOfHiddenNodes=1400, epochNumber=20, learning_rate=0.001)
        #     score, acc = twoLayerNN(numOfHiddenNodes=200, epochNumber=20, learning_rate=0.001)
        # score, acc = labelSmoothing(numOfHiddenNodes=1000, epochNumber=20, pParameter=1, learning_rate=0.001)
    #     accuracyVector.append(acc)
    #     scoreVector.append(score)
    #
    # accMean = sum(accuracyVector)*100.0/len(accuracyVector)
    # scoreMean = sum(scoreVector) * 1.0 / len(scoreVector)
    # print(' ')
    # print('ACC:', accMean)
    # print('SCORE:', scoreMean)
    # error = 0
    # for accLoop in accuracyVector:
    #     error = error + abs(accLoop - accMean)
    # errorMean = error * 1.0 / len(accuracyVector)
    # print('ERROR:', errorMean)
    # print('VAR:', np.var(accuracyVector))



    ## Distilation method
    original_acc_vector = []
    distil_acc_vector = []
    for loop in range(1, 2):
        original_acc, distil_acc = distillation(numOfHiddenNodes1=1000, numOfHiddenNodes2=100, temp=1,
                                                epochNumber=20, learnRate=0.001)
        original_acc_vector.append(original_acc)
        distil_acc_vector.append(distil_acc)

    originalMean = sum(original_acc_vector) * 100.0 / len(original_acc_vector)
    distilMean = sum(distil_acc_vector) * 100.0 / len(distil_acc_vector)
    print(' ')
    print('Original:', originalMean)
    print('Distil:', distilMean)
    print('Improvement: ', distilMean-originalMean)
        # error = 0
        # for accLoop in accuracyVector:
        #     error = error + abs(accLoop - accMean)
        # errorMean = error * 1.0 / len(accuracyVector)
        # print('ERROR:', errorMean)
        # print('VAR:', np.var(accuracyVector))


def distillation (numOfHiddenNodes1, numOfHiddenNodes2, temp, epochNumber, learnRate):
    nb_classes = 43
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    print('Size Test: ', X_test.shape)
    ## train first model
    model1 = Sequential()
    model1.add(Dense(numOfHiddenNodes1, input_shape=(28*28,)))
    model1.add(Activation('relu'))
    model1.add(Dense(nb_classes))
    model1.add(Activation('softmax'))

    optim = adam(lr=learnRate)
    model1.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    model1.fit(X_train, Y_train,
              batch_size=100, nb_epoch=epochNumber,
              show_accuracy=True, verbose=0,
              validation_data=(X_valid, Y_valid))

    Z_train = model1.predict(X_train, batch_size=100, verbose=0)
    Z_valid = model1.predict(X_valid, batch_size=100, verbose=0)
    print("Size Z: ", Z_train.shape)

    ## train second model using output of first model
    model2 = Sequential()
    model2.add(Dense(numOfHiddenNodes2, input_shape=(28 * 28,)))
    model2.add(Activation('relu'))
    model2.add(Lambda(lambda x: (x * 1.0) / temp))
    model2.add(Dense(nb_classes))
    model2.add(Activation('softmax'))


    model2.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    print(Y_train.shape)
    print(Z_train.shape)
    model2.fit(X_train, Z_train,
               batch_size=100, nb_epoch=epochNumber,
               show_accuracy=True, verbose=0,
               validation_data=(X_valid, Z_valid))


    distillation_score = model2.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=0)

    ## train third model for comparison
    model3 = Sequential()
    model3.add(Dense(numOfHiddenNodes2, input_shape=(28 * 28,)))
    model3.add(Activation('relu'))
    model3.add(Dense(nb_classes))
    model3.add(Activation('softmax'))

    model3.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    model3.fit(X_train, Y_train,
               batch_size=100, nb_epoch=epochNumber,
               show_accuracy=True, verbose=0,
               validation_data=(X_valid, Y_valid))

    original_score = model3.evaluate(X_test, Y_test,
                                         show_accuracy=True, verbose=0)
    print(len(original_score))
    print(original_score[1])
    # return original_accuracy, distilation_accuracy
    return original_score[1], distillation_score[1]

def labelSmoothing(numOfHiddenNodes, epochNumber, pParameter, learning_rate):
    nb_classes = 43
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    optim = adam(lr=learning_rate)
    # smooth labels
    prob = pParameter * 1.0 / (nb_classes - 1)
    for yIndex,dum1 in enumerate(Y_train):
        for yprob,dum2 in enumerate(Y_train[yIndex, :]):
            if (Y_train[yIndex, yprob] == 1):
                Y_train[yIndex, yprob] = 1 - pParameter
            else:
                Y_train[yIndex, yprob] = prob

    # train first model, one layer, M = 800
    model = Sequential()
    model.add(Dense(numOfHiddenNodes, input_shape=(28*28,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=100, nb_epoch=epochNumber,
              show_accuracy=True, verbose=1,
              validation_data=(X_valid, Y_valid))

    score = model.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=1)


    return score[0], score[1]

def twoLayerNN(numOfHiddenNodes, epochNumber, learning_rate):
    nb_classes = 43
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    optim = adam(lr=learning_rate)
    # train first model, one layer, M = 800
    model = Sequential()
    model.add(Dense(numOfHiddenNodes, input_shape=(28 * 28,)))
    model.add(Activation('relu'))
    model.add(Dense(numOfHiddenNodes, input_shape=(28 * 28,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=100, nb_epoch=epochNumber,
              show_accuracy=True, verbose=1,
              validation_data=(X_valid, Y_valid))

    score = model.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=1)

    return score[0], score[1]


def oneLayerNN(numOfHiddenNodes, epochNumber, learning_rate):
    nb_classes = 43
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    optim = adam(lr=learning_rate)
    # train first model, one layer, M = 800
    model = Sequential()
    model.add(Dense(numOfHiddenNodes, input_shape=(28*28,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=100, nb_epoch=epochNumber,
              show_accuracy=True, verbose=1,
              validation_data=(X_valid, Y_valid))

    score = model.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=1)
    # print("Test score:", score[0])
    # print("Test accuracy:", score[1])
    # return score, accuracy
    return score[0], score[1]

def readData():
    dirPath = os.getcwd()
    filePath = dirPath + '/train_data.csv'

    trainX = []
    trainY = []
    with open(filePath) as tsvFile:
        dataRows = csv.reader(tsvFile, delimiter='\t')
        for lineRow in dataRows:
            temp = str.split(lineRow[0], ',')
            trainX.append([float(i) for i in temp[0: 28*28]])
            trainY.append(float(temp[28*28]))


    X_train = np.array(trainX)
    Y_train = np.array(trainY)

    dirPath = os.getcwd()
    filePath = dirPath + '/test_data.csv'

    testX = []
    testY = []
    with open(filePath) as tsvFile:
        dataRows = csv.reader(tsvFile, delimiter='\t')
        for lineRow in dataRows:
            temp = str.split(lineRow[0], ',')
            testX.append([float(i) for i in temp[0: 28 * 28]])
            testY.append(float(temp[28 * 28]))

    X_test = np.array(testX)
    Y_test = np.array(testY)

    np.save(os.path.dirname(os.path.realpath(__file__)) + "/train_array_X", X_train)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "/train_array_Y", Y_train)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "/test_array_X", X_test)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "/test_array_Y", Y_test)

    print('Saving Done!')

def loadData ():
    numOfValidation = 5000;
    X_train = np.load(os.path.dirname(os.path.realpath(__file__)) + "/train_array_X.npy")
    Y_train = np.load(os.path.dirname(os.path.realpath(__file__)) + "/train_array_Y.npy")
    X_test = np.load(os.path.dirname(os.path.realpath(__file__)) + "/test_array_X.npy")
    Y_test = np.load(os.path.dirname(os.path.realpath(__file__)) + "/test_array_Y.npy")

    # divide data into train and evaluation
    X_valid = X_train[0: numOfValidation, :]
    Y_valid = Y_train[0: numOfValidation]

    return X_train[numOfValidation:, :], Y_train[numOfValidation:,], X_valid, Y_valid, X_test, Y_test


if __name__ == '__main__':
    main()