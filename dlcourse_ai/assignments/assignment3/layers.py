import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.square(W))
    return loss, 2 * reg_strength * W


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    def row_softmax(row):
        exp_row = np.exp(row - np.max(row))
        sum = np.sum(exp_row)
        return exp_row / sum

    if len(predictions.shape) == 1:
        probs = row_softmax(predictions)
    else:
        probs = np.zeros_like(predictions, dtype=float)
        for row_idx in range(len(predictions)):
            probs[row_idx] = row_softmax(predictions[row_idx])

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    if len(probs.shape) == 1:
        probs = probs[np.newaxis]

    loss = 0.0

    for row_idx in range(len(probs)):
        row_probs = probs[row_idx]
        class_idx = target_index[row_idx]
        loss -= np.log(row_probs[class_idx])

    return loss / len(probs)


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = np.copy(probs)
    assert predictions.shape == dprediction.shape, (predictions.shape, dprediction.shape)

    def row_dprediction(dprediction_row, row_target_index):
        dprediction_row[row_target_index] -= 1

    if len(dprediction.shape) == 1:
        row_dprediction(dprediction, target_index[0])
    else:
        for row_idx in range(len(dprediction)):
            row_dprediction(dprediction[row_idx], target_index[row_idx])

    return loss, dprediction / target_index.shape[0]



class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.indicies = np.nonzero(X <= 0)
        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_out[self.indicies] = 0
        return d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, x_height, x_width, channels = X.shape

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        p = self.padding

        out_height = (x_height - self.filter_size + 2 * p) + 1
        out_width = (x_width - self.filter_size + 2 * p) + 1

        x_padded = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), "constant")
        output = np.zeros((batch_size, out_height, out_width, self.out_channels), dtype=float)
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                output[:, y, x, :] = np.sum(x_padded[:, y: y + self.filter_size, x: x + self.filter_size, :, np.newaxis] * self.W.value, axis=(1, 2, 3)) + self.B.value

        return output

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        p = self.padding
        X = self.X
        x_padded = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), "constant")
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        dX = np.zeros_like(x_padded, dtype=float)
        self.W.grad.fill(0.0)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                dX[:, y: y + self.filter_size, x: x + self.filter_size] \
                    += np.sum(self.W.value[np.newaxis, ...] * d_out[:, y: y + 1, x: x + 1, np.newaxis, :], axis=4)

                self.W.grad += np.sum(x_padded[:, y: y + self.filter_size, x: x + self.filter_size, :, np.newaxis] * d_out[:, y: y + 1, x: x + 1, np.newaxis, :], axis=0)

        self.B.grad = np.sum(d_out, axis=(0, 1, 2))

        if p != 0:
            dX = dX[:, p:-p, p:-p, :]

        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        stride = self.stride
        psz = self.pool_size

        assert (height - psz) % stride == 0, f"Incompatible X height {height}, pool size {psz} and stride {stride}"
        assert (width - psz) % stride == 0, f"Incompatible X width {width}, pool size {psz} and stride {stride}"

        self.X = X.copy()
        out_height = (height - psz) // stride + 1
        out_width = (width - psz) // stride + 1
        output = np.zeros((batch_size, out_height, out_width, channels), dtype=float)

        for y in range(out_height):
            for x in range(out_width):
                np.amax(X[:, y * stride: y * stride + psz, x * stride: x * stride + psz, :], axis=(1, 2), out=output[:, y, x, :])

        return output


    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        stride = self.stride
        psz = self.pool_size

        dX = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                flat_max_idx = np.argmax(self.X[:, y * stride: y * stride + psz, x * stride: x * stride + psz, :]
                                         .reshape(batch_size, psz * psz, channels), axis=1)
                (iy, ix) = np.unravel_index(flat_max_idx.ravel(), (psz, psz))

                idxs = tuple(np.array([(bs, iy[bs * channels + ch], ix[bs * channels + ch], ch)
                                       for bs in range(batch_size) for ch in range(channels)]).T)

                dX[:, y * stride: y * stride + psz, x * stride: x * stride + psz, :][idxs] += d_out[:, y, x, :].ravel()

        return dX


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
