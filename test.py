import numpy as np
import pickle
from network import predict

np.set_printoptions(suppress=True, precision=4)

with open("network.pkl", "rb") as f:
    network = pickle.load(f)

televisionX = np.round(np.load("television.npy") / 255, 3)
televisionY = np.array([[1, 0, 0, 0, 0]]*50000)
televisionX_test = np.round(np.load("television_test.npy") / 255, 3)
televisionY_test = np.array([[1, 0, 0, 0, 0]]*1000)

bowtieX = np.round(np.load("bowtie.npy") / 255, 3)
bowtieY = np.array([[0, 1, 0, 0, 0]]*50000)
bowtieX_test = np.round(np.load("bowtie_test.npy") / 255, 3)
bowtieY_test = np.array([[0, 1, 0, 0, 0]]*1000)

basketballX = np.round(np.load("basketball.npy") / 255, 3)
basketballY = np.array([[0, 0, 1, 0, 0]]*50000)
basketballX_test = np.round(np.load("basketball_test.npy") / 255, 3)
basketballY_test = np.array([[0, 0, 1, 0, 0]]*1000)

donutX = np.round(np.load("donut.npy") / 255, 3)
donutY = np.array([[0, 0, 0, 1, 0]]*50000)
donutX_test = np.round(np.load("donut_test.npy") / 255, 3)
donutY_test = np.array([[0, 0, 0, 1, 0]]*1000)

mushroomX = np.round(np.load("mushroom.npy") / 255, 3)
mushroomY = np.array([[0, 0, 0, 0, 1]]*50000)
mushroomX_test = np.round(np.load("mushroom_test.npy") / 255, 3)
mushroomY_test = np.array([[0, 0, 0, 0, 1]]*1000)

X = np.concatenate((televisionX, bowtieX, basketballX, donutX, mushroomX))
Y = np.concatenate((televisionY, bowtieY, basketballY, donutY, mushroomY))
X = np.reshape(X, (250000, 784, 1))
Y = np.reshape(Y, (250000, 5, 1))

X_test = np.concatenate((televisionX_test, bowtieX_test, basketballX_test, donutX_test, mushroomX_test))
Y_test = np.concatenate((televisionY_test, bowtieY_test, basketballY_test, donutY_test, mushroomY_test))
X_test = np.reshape(X_test, (5000, 784, 1))
Y_test = np.reshape(Y_test, (5000, 5, 1))

def calculate_accuracy(network, X, Y, dataset_type):
    correct = 0
    total = 0
    for x, y in zip(X, Y):
        output = predict(network, x)
        if (np.argmax(output) == np.argmax(y)):
            correct += 1
        total += 1
    accuracy = correct/total
    print(f"{dataset_type} accuracy: {'{:.2f}'.format(round(accuracy*100, 4))}%")
    return accuracy

train_accuracy = calculate_accuracy(network, X, Y, "training")
test_accuracy = calculate_accuracy(network, X_test, Y_test, "test")