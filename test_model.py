import saber.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Inputs: 2, 400
        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        X = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.sigmoid(x)

        assert(out.shape == (1, X.shape[1]))
        return out
