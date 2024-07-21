import numpy as np

def f2_score(y_true, y_pred, threshold=0.5):
    """
    Calculate the F2 score based on predictions and ground truth.
    """
    # Flatten the arrays
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # Threshold the predictions
    y_pred_f = (y_pred_f > threshold).astype(int)

    # Calculate true positives, false negatives, and false positives
    TP = np.sum(y_true_f * y_pred_f)
    FP = np.sum((1 - y_true_f) * y_pred_f)
    FN = np.sum(y_true_f * (1 - y_pred_f))

    # Calculate F2 score
    if TP == 0:
        return 0
    return (5 * TP) / (5 * TP + 4 * FN + FP)
