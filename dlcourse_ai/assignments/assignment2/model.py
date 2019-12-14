import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, softmax, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fcl1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fcl2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.layers = [self.fcl1, self.relu1, self.fcl2]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param in self.params().values():
            param.grad.fill(0)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        output_data = X.copy()
        for layer in self.layers:
            output_data = layer.forward(output_data)

        loss, dpredictions = softmax_with_cross_entropy(output_data, y)

        doutput_data = dpredictions
        for layer in reversed(self.layers):
            doutput_data = layer.backward(doutput_data)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            l2_loss, dparam = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += dparam

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        output_data = X
        for layer in self.layers:
            output_data = layer.forward(output_data)

        probs = softmax(output_data)
        return np.argmax(probs, axis=1)

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for (key, value) in self.fcl1.params().items():
            result[f"{key}1"] = value

        for (key, value) in self.fcl2.params().items():
            result[f"{key}2"] = value

        return result
