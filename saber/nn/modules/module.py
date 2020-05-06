from collections import OrderedDict


class Module(object):
    """ 
    This will be used as a base class for all the Neural Network modules.

    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *input):
        # This file has NotImplementedError but
        # will be overwritten by all the classes
        # that use this module.
        raise NotImplementedError
