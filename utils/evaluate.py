from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _split_items(csv_cell: str) -> List[str]:
    if not isinstance(csv_cell, str) or not csv_cell.strip():
        return []
    return [p.strip() for p in csv_cell.split(",") if p.strip()]


def evaluate_recommender(df: pd.DataFrame, get_recommendations_fn) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 for the rule-based recommender.

    For each row in the dataset, we generate labeled examples for foods in
    'Recommended Foods' (label 1) and 'Foods/Ingredients Not Recommended'
    (label 0). We compare our function's output against these labels.
    Unknown predictions are treated as negative (0).
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    for _, row in df.iterrows():
        disease = str(row.get("Disease", "")).strip()
        rec_items = _split_items(str(row.get("Recommended Foods", "")))
        not_rec_items = _split_items(str(row.get("Foods/Ingredients Not Recommended", "")))

        for food in rec_items:
            preds = get_recommendations_fn(disease, [food])
            pred_label = 1 if preds and preds[0].get("Recommendation") == "Recommended" else 0
            y_true.append(1)
            y_pred.append(pred_label)

        for food in not_rec_items:
            preds = get_recommendations_fn(disease, [food])
            pred_label = 1 if preds and preds[0].get("Recommendation") == "Recommended" else 0
            y_true.append(0)
            y_pred.append(pred_label)

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(len(y_true)),
    }

