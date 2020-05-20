from collections import OrderedDict
import numpy as np

# TODO: Implement the model parameters function for Adam Optimizer


class Module(object):
    """ 
    This will be used as a base class for all the Neural Network modules.

    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *inputs):
        # This file has NotImplementedError but
        # will be overwritten by all the classes
        # that use this module.
        raise NotImplementedError

    def backwards(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        """This allows the object to be callable.
        For example, you can define a model like follows

            def __init__(self):
                self.fc1 = nn.Linear(inputs, outputs)

            def forward(self, inputs):
                inputs = self.fc1(inputs):

        The forward is run automatically        
        """
        result = self.forward(*inputs, **kwargs)

        return result
