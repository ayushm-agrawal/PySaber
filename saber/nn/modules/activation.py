from . import Module
import numpy as np


class ReLU(Module):
    """ ReLU or Rectified Linear Unit activation function.
        Defined as the positive part of its argument

        f(x) = x+ = max(0, x)

    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return np.max(0, input)


class Sigmoid(Module):
    """ Sigmoid Activation Function.

        A sigmoid function is a bounded, differentiable, real function that is 
        defined for all real input values and has a non-negative derivative at each point.

        sig(x) =    1
                ---------
                1 + e^(-x)

    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        return (1/(1+np.exp(-input)))


class Leaky_ReLU(Module):
    raise NotImplementedError
