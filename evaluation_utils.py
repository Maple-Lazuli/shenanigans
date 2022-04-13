import pandas as pd
import numpy as np


def calc_precision(tp, fp):
    return tp / (tp + fp) if tp != 0 else 0


def calc_recall(tp, fn):
    return tp / (tp + fn) if tp != 0 else 0


def calc_specificity(tn, fp):
    return tn / (tn + fp)


def calc_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn) if tp != 0 or tn != 0 else 0


def calc_f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp != 0 else 0


def confusion_matrix_calculations(matrix):
    scores = []

    for label in range(0, matrix.shape[0]):

        # find the true positive
        tp = matrix.iloc[label, label]

        # find the true negative
        tn = 0
        for row in range(0, matrix.shape[0]):
            if row == label:
                continue
            for col in range(0, matrix.shape[1]):
                if col == label:
                    continue
                tn += matrix.iloc[row, col]

        # find the false positive
        fp = 0
        for col in range(0, matrix.shape[1]):
            if col == label:
                continue
            fp += matrix.iloc[label, col]

        fn = 0
        for row in range(0, matrix.shape[0]):
            if row == label:
                continue
            fn += matrix.iloc[row, label]

        label_calculation_dict = {"label": label,
                                  "true-pos": tp,
                                  "true-neg": tn,
                                  "false-pos": fp,
                                  "false-neg": fn,
                                  "precision": calc_precision(tp=tp, fp=fp),
                                  "recall": calc_recall(tp=tp, fn=fn),
                                  "specificity": calc_specificity(tn=tn, fp=fp),
                                  "misclassification rate": 1 - calc_specificity(tn=tn, fp=fp),
                                  "accuracy": calc_accuracy(tp=tp, tn=tn, fp=fp, fn=fn),
                                  "f1-score": calc_f1_score(tp=tp, fp=fp, fn=fn),
                                  "tpr": float(tp) / float(tp + fn),
                                  "fpr": float(fp) / float(fp + tn)
                                  }

        scores.append(label_calculation_dict)

    return scores


def create_measure_matrix(confusion_matrix):
    """
    Creates a pandas dataframe of the scores from a confusion matrix

    Parameters
    ----------
    confusion_matrix

    Returns
    -------

    """
    scores = []
    for row in range(0, confusion_matrix.shape[0]):
        scores.append(confusion_matrix_calculations_for_label(row, row, confusion_matrix))

    score_matrix = pd.DataFrame.from_dict(scores)

    return score_matrix


def confusion_matrix_calculations_for_label(prediction_idx, true_idx, matrix):
    # find the true positive
    tp = matrix.iloc[prediction_idx, true_idx]

    # find the true negative
    tn = 0
    for row in range(0, matrix.shape[0]):
        if row == prediction_idx:
            continue
        for col in range(0, matrix.shape[1]):
            if col == true_idx:
                continue
            tn += matrix.iloc[row, col]

    # find the false positive
    fp = 0
    for col in range(0, matrix.shape[1]):
        if col == true_idx:
            continue
        fp += matrix.iloc[prediction_idx, col]

    fn = 0
    for row in range(0, matrix.shape[0]):
        if row == prediction_idx:
            continue
        fn += matrix.iloc[row, true_idx]

    label_calculation_dict = {"label": prediction_idx,
                              "true-pos": tp,
                              "true-neg": tn,
                              "false-pos": fp,
                              "false-neg": fn,
                              "precision": calc_precision(tp=tp, fp=fp),
                              "recall": calc_recall(tp=tp, fn=fn),
                              "specificity": calc_specificity(tn=tn, fp=fp),
                              "misclassification rate": 1 - calc_specificity(tn=tn, fp=fp),
                              "accuracy": calc_accuracy(tp=tp, tn=tn, fp=fp, fn=fn),
                              "f1-score": calc_f1_score(tp=tp, fp=fp, fn=fn)

                              }

    return label_calculation_dict


def expand_lists(x_list, y_list):
    """
    Doubles the size of the two lists for the line plot in the ROC Curves

    Parameters
    ----------
    x_list: Values along the horizontal axis
    y_list: Values along the vertical axis

    Returns
    -------
    Returns a doubled up version of both lists.
    """
    x_expanded = [x_list[0]]
    y_expanded = []

    for x in x_list[1:]:
        x_expanded.append(x)
        x_expanded.append(x)

    for y in y_list[:-1]:
        y_expanded.append(y)
        y_expanded.append(y)

    y_expanded.append(y_list[-1])

    return x_expanded, y_expanded


def create_ovr_roc_dict(classifications, num_labels, resolution=100):
    """
    Create ROC curves for each label using the one versus rest appraoch.
    Parameters
    ----------
    classifications: an np.ndarray consisting of the predictions for a class
    num_labels: a list of the possible labels in the class. expected to be a list of integers.
    resolution: the number of steps between 0 and 1 for the ROC curve

    Returns
    -------
    Dictionary with one key for each label. Each key is mapped to two lists, each containing the true positive rate and
    the false positive rate
    """
    # prepare the ROC dictionary

    labels = list(range(0, num_labels))

    roc_dict = {}
    for label in labels:
        roc_dict[label] = {
            'tp_rates': [],
            'fp_rates': []
        }

    for threshold in np.linspace(0, 1, num=resolution):

        # for each label, find one versus rest
        for label in labels:
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0

            # for each prediction in predictions from the model iterating through the validation set
            for classification in classifications:

                # determine if the label is the true classification of the image
                # Note: the label is appended to the end of each classification
                if label == int(classification[-1]):
                    label_parity = True
                else:
                    label_parity = False

                # retrieve the probability of the label
                probability_of_label = classification[label]

                # if (probability_of_label >= threshold) and (probability_of_label > 1 - probability_of_label): # is this the proper way to do this part?
                if (probability_of_label >= threshold) and (probability_of_label > 1 - probability_of_label):
                    classifier_parity = True
                else:
                    classifier_parity = False

                if label_parity:
                    if classifier_parity:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if classifier_parity:
                        false_positives += 1
                    else:
                        true_negatives += 1

            true_positive_rate = true_positives / (true_positives + false_negatives) if true_positives != 0 else 0
            false_positive_rate = false_positives / (false_positives + true_negatives) if false_positives != 0 else 0

            roc_dict[label]['tp_rates'].append(true_positive_rate)
            roc_dict[label]['fp_rates'].append(false_positive_rate)

    for label in labels:
        lagged_x, lagged_y = expand_lists(roc_dict[label]['fp_rates'], roc_dict[label]['tp_rates'])
        roc_dict[label]['fp_rates'] = lagged_x
        roc_dict[label]['tp_rates'] = lagged_y

    return roc_dict


def create_confusion_matrix(classifications, labels):
    # create confusion matrix
    num_labels = len(labels)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=np.int)
    column_names = ["true_" + str(label) for label in labels]
    row_names = ["predicted_" + str(label) for label in labels]
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=column_names)
    confusion_matrix.index = row_names
    # fill confusion matrix
    for classification in classifications:
        true_class = int(classification[-1])
        predicted_class = np.argmax(classification[:-1])
        confusion_matrix.iloc[predicted_class, true_class] += 1

    return confusion_matrix
