import numpy as np


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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    loss = reg_strength * np.sum(np.square(W))
    return loss, 2 * reg_strength * W
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = X.dot(W)

    # TODO implement prediction and gradient over W
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    return loss, X.T.dot(dpredictions)


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            loss = 0.0

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for batch_indices in batches_indices:
                batch = X[batch_indices]
                batch_labels = y[batch_indices]
                linear_loss, dlinear_loss = linear_softmax(batch, self.W, batch_labels)
                regul_loss, dregul_loss = l2_regularization(self.W, reg)
                self.W -= learning_rate * (dlinear_loss + dregul_loss)
                loss = linear_loss * regul_loss

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        # TODO Implement class prediction
        return np.argmax(X.dot(self.W), axis=1)



                
                                                          

            

                
