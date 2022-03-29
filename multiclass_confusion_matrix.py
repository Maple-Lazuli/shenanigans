import pandas as pd
import numpy as np


def calc_precision(tp, fp):
    return tp / (tp + fp)


def calc_recall(tp, fn):
    return tp / (tp + fn)


def calc_specificity(tn, fp):
    return tn / (tn + fp)


def calc_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def calc_f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def confusion_matrix_calculations(prediction_idx, true_idx, matrix):
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


def create_measure_matrix(confusion_matrix):
    scores = []
    for row in range(0, confusion_matrix.shape[0]):
        scores.append(confusion_matrix_calculations(row, row, confusion_matrix))

    score_matrix = pd.DataFrame.from_dict(scores)

    return score_matrix


def create_confusion_matrix(labels):
    confusion_matrix = np.zeros((10, 10), dtype=np.int)
    column_names = ["true_" + str(label) for label in labels]
    row_names = ["predicted_" + str(label) for label in labels]
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=column_names)
    confusion_matrix.index = row_names
    return confusion_matrix
