"""Step 7: evaluation metrics — structure quality + generation quality.

Structure (vs. ground-truth DAG): SHD, edge precision/recall/F1, final BIC.
Generation: held-out log-likelihood, MMD/JS across marginals, posterior
predictive checks (empirical vs. simulated moments/histograms).
"""

import numpy as np


def shd(A_pred, A_true):
    """Structural Hamming Distance: # of edge additions/deletions/reversals to
    turn A_pred into A_true. This one is implemented so evaluate.py is usable."""
    A_pred, A_true = np.asarray(A_pred), np.asarray(A_true)
    diff = 0
    d = len(A_true)
    for i in range(d):
        for j in range(i + 1, d):
            p = (A_pred[i, j], A_pred[j, i])
            t = (A_true[i, j], A_true[j, i])
            if p != t:
                diff += 1  # any mismatch on this pair (missing/extra/reversed) = 1
    return diff


def precision_recall_f1(A_pred, A_true):
    """Directed-edge precision, recall, F1."""
    A_pred, A_true = np.asarray(A_pred), np.asarray(A_true)
    tp = int(((A_pred == 1) & (A_true == 1)).sum())
    pred_pos = int((A_pred == 1).sum())
    true_pos = int((A_true == 1).sum())
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / true_pos if true_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _demo():
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0->1->2
    perfect = true.copy()
    assert shd(perfect, true) == 0
    assert precision_recall_f1(perfect, true) == (1.0, 1.0, 1.0)

    reversed_edge = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])  # 1->0 instead of 0->1
    assert shd(reversed_edge, true) == 1  # one pair differs
    print("evaluate.py self-check passed")


if __name__ == "__main__":
    _demo()
