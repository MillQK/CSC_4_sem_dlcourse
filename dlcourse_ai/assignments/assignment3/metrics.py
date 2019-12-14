def binary_classification_metrics(prediction, ground_truth):

    # TODO: implement metrics!
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

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    assert len(prediction) == len(ground_truth)

    count = len(prediction)
    accuracy = 0

    for i in range(count):
        if prediction[i] == ground_truth[i]:
            accuracy += 1

    if count != 0:
        accuracy /= count

    return accuracy
