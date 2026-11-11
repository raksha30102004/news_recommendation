"""Per-impression ranking metrics, matching the official MIND evaluation script (report §4.4)."""
import numpy as np
from scipy.stats import rankdata


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y = y_true[order]
    return float(np.sum(y / (np.arange(len(y)) + 1)) / max(np.sum(y_true), 1))


def _dcg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1]
    y = y_true[order[:k]]
    return float(np.sum((2 ** y - 1) / np.log2(np.arange(len(y)) + 2)))


def ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    best = _dcg(y_true, y_true, k)
    return _dcg(y_true, y_score, k) / best if best > 0 else 0.0


def evaluate(labels: list[np.ndarray], scores: list[np.ndarray]) -> dict:
    rows = np.array([[auc(y, s), mrr(y, s), ndcg(y, s, 5), ndcg(y, s, 10)] for y, s in zip(labels, scores)])
    m = np.nanmean(rows, axis=0)
    return {"auc": round(float(m[0]), 4), "mrr": round(float(m[1]), 4), "ndcg@5": round(float(m[2]), 4), "ndcg@10": round(float(m[3]), 4)}
