import saber.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        X = x
        x = self.fc1.forward(x)
        print("After first Linear Layer")
        print("Shape: {}...".format(x.shape))
        x = self.relu.forward(x)
        print("After ReLU")
        print("Shape: {}...".format(x.shape))
        x = self.fc2.forward(x)
        print("After second Linear Layer")
        print("Shape: {}...".format(x.shape))
        out = self.sigmoid.forward(x)
        print("Output Sigmoid Layer")
        print("Shape: {}".format(out.shape))
        assert(out.shape == (1, X.shape[1]))
        return out
