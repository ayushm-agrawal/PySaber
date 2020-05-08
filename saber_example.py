import matplotlib.pyplot as plt
import numpy as np

import saber.nn as nn
from test_dataset import load_planar_dataset, plot_decision_boundary
from test_model import Net
import saber.optim as optim


def main():
    # This function implements a Neural Network using Saber
    X, Y = load_planar_dataset()

    # save the data plot in a file
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.savefig('./data.png')

    # load the custom model
    model = Net()

    # intialize the loss function
    criterion = nn.CrossEntropyLoss()

    # TODO: Implement Adam Optimizer
    # optimizer = optim.Adam(model._parameters, lr=0.003)

    # Training loop
    epochs = 10

    for epoch in range(epochs):
        # Forward pass through the model
        output = model(X)

        # Calculate the loss
        loss = criterion(output, Y)
        print(loss)
        print("-----")


if __name__ == '__main__':
    main()
