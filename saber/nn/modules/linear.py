from .module import Module
import numpy as np


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.input = in_features
        self.output = out_features
        self.weights = np.random.randn(out_features, in_features) * 0.01

        if bias:
            self.bias = self.set_bias()
            assert(self.bias.shape == (self.output, 1))

    def set_bias(self):
        # set the bias vector using a uniform distribution
        # Bounds for the uniform distribution are derived using Kaiming Initialization

        bound_range = np.sqrt((2/self.input))
        bias = np.random.uniform(-bound_range, bound_range, self.output)
        return np.reshape(bias, (bias.shape[0], 1))

    def forward(self, input):
        print(input.shape, self.weights.T.shape, self.bias.shape)
        # output = np.matmul(input, self.weights.T) + self.bias
        output = np.dot(self.weights, input) + self.bias
        return output
