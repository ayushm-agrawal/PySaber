from .module import Module
import numpy as np


class NLLLoss(Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, input, target):
        # TODO: NLLLoss is missing forward implementation
        pass


class CrossEntropyLoss(Module):
    """The Cross Entropy Loss.
    It is a measure of the difference between two probability distribution for a given
    random variable or set of events.

    Entropy is the number of bits required to transmit a randomly selected event from a 
    probability distribution. A skewed distribution has low entropy, whereas a distribution
    where events have equal probability has a larger entropy.

    Args:
        input: Predicted outputs from the model: Shape (1, num_examples)
        target: True labels for the dataset: Shape(1, num_examples)
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        pass

    def forward(self, input, target):

        num_examples = target.shape[1]

        cost = - np.sum(np.multiply(target, np.log(input) +
                                    np.multiply((1-target), np.log((1-input)))))

        cost = cost/num_examples

        # convert cost to a scalar
        cost = np.squeeze(cost)

        assert(cost.shape == ())

        return cost
