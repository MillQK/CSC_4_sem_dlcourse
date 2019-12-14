def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    assert len(prediction) == len(ground_truth)

    count = len(prediction)
    accuracy = 0

    for i in range(count):
        if prediction[i] == ground_truth[i]:
            accuracy += 1

    if count != 0:
        accuracy /= count

    return accuracy
