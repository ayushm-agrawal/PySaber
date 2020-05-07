import saber.nn as nn
from test_model import Net
from test_dataset import load_planar_dataset, plot_decision_boundary
import matplotlib.pyplot as plt
import numpy as np


def main():
    # This function implements a Neural Network using Saber
    X, Y = load_planar_dataset()

    # save the data plot in a file
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.savefig('./data.png')

    model = Net()
    output = model.forward(X)
    print(output.shape)


if __name__ == '__main__':
    main()
