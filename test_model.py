import saber.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.fc1(x)
        out = self.fc2(x)

        return out
