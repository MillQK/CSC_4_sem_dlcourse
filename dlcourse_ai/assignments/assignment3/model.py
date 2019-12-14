import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv1 = ConvolutionalLayer(in_channels=input_shape[-1], out_channels=conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(pool_size=4, stride=4)
        self.conv2 = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(pool_size=4, stride=4)
        self.flatten1 = Flattener()
        self.fcl1 = FullyConnectedLayer(n_input=2*2*conv2_channels, n_output=n_output_classes)

        self.layers = [self.conv1, self.relu1, self.maxpool1,
                       self.conv2, self.relu2, self.maxpool2,
                       self.flatten1, self.fcl1]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param in self.params().values():
            param.grad.fill(0.0)

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        output_data = X.copy()
        for layer in self.layers:
            output_data = layer.forward(output_data)

        loss, dpredictions = softmax_with_cross_entropy(output_data, y)

        doutput_data = dpredictions
        for layer in reversed(self.layers):
            doutput_data = layer.backward(doutput_data)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        output_data = X
        for layer in self.layers:
            output_data = layer.forward(output_data)

        probs = softmax(output_data)
        return np.argmax(probs, axis=1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for idx, layer in enumerate(self.layers):
            for key, value in layer.params().items():
                result[f"{key}{idx}"] = value

        return result
