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
        return np.maximum(0, input)

    def backwards(self, dA, cache):
        """
        Implement back prop for a single ReLU unit.
        Args:
        dA -- post-activation gradient, of any shape
        cache -- used to store output for back prop computation
        """

        dZ = np.array(dA, copy=True)

        dZ[cache <= 0] = 0

        assert (dZ.shape == cache.shape)

        return dZ


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
        return 1/(1+np.exp(-input))

    def backwards(self, dA, cache):
        """
        Implement back prop for a single Sigmoid unit.
        Args:
        dA -- post-activation gradient, of any shape
        cache -- used to store output for back prop computation
        """
        s = self.forward(cache)

        dZ = dA * s * (1-s)

        return dZ
