import numpy as np
from activation import Activation

class Sigmoid(Activation):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)