from .module import Module


class NLLLoss(Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, input, target):
        # TODO: Implement this later
        pass


class CrossEntropyLoss(Module):
    """The Cross Entropy Loss.
    It is a measure of the difference between two probability distribution for a given
    random variable or set of events.

    Entropy is the number of bits required to transmit a randomly selected event from a 
    probability distribution. A skewed distribution has low entropy, whereas a distribution
    where events have equal probability has a larger entropy.

    Args:
        weight (Tensor, optional): 
    """

    def __init__(self, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        raise NotImplementedError
