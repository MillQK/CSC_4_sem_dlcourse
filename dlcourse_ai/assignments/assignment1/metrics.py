def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    assert len(prediction) == len(ground_truth)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(prediction)):
        if prediction[i]:
            if ground_truth[i]:
                tp += 1
            else:
                fp += 1
        else:
            if ground_truth[i]:
                fn += 1
            else:
                tn += 1

    accuracy = (tp + tn)/(tp + tn + fp + fn)

    if fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)

    if fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    f1 = 2 * tp / (2 * tp + fn + fp)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    assert len(prediction) == len(ground_truth)

    count = len(prediction)
    accuracy = 0

    for i in range(count):
        if prediction[i] == ground_truth[i]:
            accuracy += 1

    return accuracy / count
