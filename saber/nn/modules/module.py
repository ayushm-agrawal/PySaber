from collections import OrderedDict


class Module(object):
    """ 
    This will be used as a base class for all the Neural Network modules.

    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *input):
        raise NotImplementedError
