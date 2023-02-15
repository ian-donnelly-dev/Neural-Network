import numpy as np
import pickle

from dense import Dense
from sigmoid import Sigmoid
from error import mse, mse_prime
from network import train
from shuffle import unison_shuffle

np.set_printoptions(suppress=True, precision=4)

televisionX = np.round(np.load("television.npy") / 255, 3)
televisionY = np.array([[1, 0, 0, 0, 0]]*50000)

bowtieX = np.round(np.load("bowtie.npy") / 255, 3)
bowtieY = np.array([[0, 1, 0, 0, 0]]*50000)

basketballX = np.round(np.load("basketball.npy") / 255, 3)
basketballY = np.array([[0, 0, 1, 0, 0]]*50000)

donutX = np.round(np.load("donut.npy") / 255, 3)
donutY = np.array([[0, 0, 0, 1, 0]]*50000)

mushroomX = np.round(np.load("mushroom.npy") / 255, 3)
mushroomY = np.array([[0, 0, 0, 0, 1]]*50000)

X = np.concatenate((televisionX, bowtieX, basketballX, donutX, mushroomX))
Y = np.concatenate((televisionY, bowtieY, basketballY, donutY, mushroomY))

X = np.reshape(X, (250000, 784, 1))
Y = np.reshape(Y, (250000, 5, 1))

X, Y = unison_shuffle(X, Y)

network = [
    Dense(784, 15),
    Sigmoid(),
    Dense(15, 15),
    Sigmoid(),
    Dense(15, 5),
    Sigmoid(),
]

train(network, mse, mse_prime, X, Y, epochs=10, learning_rate=0.035)
print("done training")

input("press enter to save network as pickle (will overwrite)")
with open("network.pkl", "wb") as f:
    pickle.dump(network, f)