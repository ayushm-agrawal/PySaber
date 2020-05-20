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
        self.input = input
        self.target = target
        return cost

    def backwards(self, cache):
        grads = {}
        num_layers = len(cache)

        num_examples = self.input.shape[1]

        # make target the same shape as output layer
        self.target = self.target.reshape(self.input.shape)

        dOutput = -(np.divide(self.target, self.input) -
                    np.divide(1 - self.target, 1 - self.input))

    def __backward__(self, grad_cost, W, b, A_prev):
        num_examples = input.shape[1]

        dW = np.dot(grad_cost, input.T)/num_examples
        db = np.sum(grad_cost, axis=1, keepdims=True)/num_examples
        dA_prev = np.dot(W.T, grad_cost)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        single_step_dict = {"dA_prev": dA_prev,
                            "dW": dW,
                            "db": db}

        return single_step_dict

#  def L_model_backward(AL, Y, caches):
#     """
#     Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

#     Arguments:
#     AL -- probability vector, output of the forward propagation (L_model_forward())
#     Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
#     caches -- list of caches containing:
#                 every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
#                 the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

#     Returns:
#     grads -- A dictionary with the gradients
#              grads["dA" + str(l)] = ...
#              grads["dW" + str(l)] = ...
#              grads["db" + str(l)] = ...
#     """
#     grads = {}
#     L = len(caches) # the number of layers
#     m = AL.shape[1]
#     Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

#     # Initializing the backpropagation
#     ### START CODE HERE ### (1 line of code)
#     dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
#     ### END CODE HERE ###

#     # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
#     ### START CODE HERE ### (approx. 2 lines)
#     current_cache = caches[-1]
#     grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
#     ### END CODE HERE ###

#     # Loop from l=L-2 to l=0
#     for l in reversed(range(L-1)):
#         # lth layer: (RELU -> LINEAR) gradients.
#         # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
#         ### START CODE HERE ### (approx. 5 lines)
#         current_cache = caches[l]
#         dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
#         grads["dA" + str(l)] = dA_prev_temp
#         grads["dW" + str(l + 1)] = dW_temp
#         grads["db" + str(l + 1)] = db_temp
#         ### END CODE HERE ###

#     return grads
