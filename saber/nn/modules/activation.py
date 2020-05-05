from . import Module
import numpy as np


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return np.max(0, input)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        # TODO: implement foward pass for Sigmoid Activation
        pass
