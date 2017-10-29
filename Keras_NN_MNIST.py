
from keras.datasets import mnist
from keras.optimizers import adam
from keras.layers import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np

def main():

    # accuracyVector = []
    # scoreVector = []
    # for loop in range(1, 2):
    #     # score, acc = oneLayerNN(numOfHiddenNodes=1200, epochNumber=20, learning_rate=0.001)
    #     # score, acc = twoLayerNN(numOfHiddenNodes=800, epochNumber=20, learning_rate=0.001)
    #     score, acc = labelSmoothing(numOfHiddenNodes=800, epochNumber=20, pParameter=0.9, learning_rate=0.001 )
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

    # Distilation method
    original_acc_vector = []
    distil_acc_vector = []
    for loop in range(1, 2):
        for loop in range(1, 10):
            original_acc, distil_acc = distillation(numOfHiddenNodes1=1200, numOfHiddenNodes2=100, temp=10,
                                                    epochNumber=20, learning_rate=0.001)
            original_acc_vector.append(original_acc)
            distil_acc_vector.append(distil_acc)

        originalMean = sum(original_acc_vector) * 100.0 / len(original_acc_vector)
        distilMean = sum(distil_acc_vector) * 100.0 / len(distil_acc_vector)
        print(' ')
        print('Original:', originalMean)
        print('Distil:', distilMean)
        # error = 0
        # for accLoop in accuracyVector:
        #     error = error + abs(accLoop - accMean)
        # errorMean = error * 1.0 / len(accuracyVector)
        # print('ERROR:', errorMean)
        # print('VAR:', np.var(accuracyVector))


def distillation (numOfHiddenNodes1, numOfHiddenNodes2, temp, epochNumber, learning_rate):
    nb_classes = 10
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)


    optim = adam(lr=learning_rate)

    ## train first model
    model1 = Sequential()
    model1.add(Dense(numOfHiddenNodes1, input_shape=(28*28,)))
    model1.add(Activation('relu'))
    model1.add(Dense(nb_classes))
    model1.add(Activation('softmax'))

    optim = adam(lr=learning_rate)
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
    model3.add(Dense(10))
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
    nb_classes = 10
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
    # print("Test score:", score[0])
    # print("Test accuracy:", score[1])
    # return score, accuracy
    return score[0], score[1]

def twoLayerNN(numOfHiddenNodes, epochNumber, learning_rate):
    nb_classes = 10
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
    # print("Test score:", score[0])
    # print("Test accuracy:", score[1])
    # return score, accuracy
    return score[0], score[1]


def oneLayerNN(numOfHiddenNodes, epochNumber, learning_rate):
    nb_classes = 10
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    optim = adam(lr=learning_rate)

    # train first model, one layer, M = 800
    model = Sequential()
    model.add(Dense(numOfHiddenNodes, input_shape=(28*28, )))
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

def loadData():
    numOfValidation = 10000;
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(len(X_train), 28 * 28)
    X_test = X_test.reshape(len(X_test), 28 * 28)
    # normalize data to [0,1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_valid = X_train[0: numOfValidation, :]
    Y_valid = Y_train[0: numOfValidation]

    # print(X_train[numOfValidation:, :].shape)
    # print(Y_train[numOfValidation:, ].shape)
    #
    # print(X_valid.shape)
    # print(Y_valid.shape)
    #
    # print(X_test.shape)
    # print(Y_test.shape)

    return X_train[numOfValidation:, :], Y_train[numOfValidation:, ], X_valid, Y_valid, X_test, Y_test

if __name__ == '__main__':
    main()