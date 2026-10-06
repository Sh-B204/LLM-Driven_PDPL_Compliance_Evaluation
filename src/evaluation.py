"""Notebook evaluation helpers preserved for LATER analysis (cell 10). NOT executed by the pipeline:
this task is about reliable execution and result preservation; no new statistics are computed here."""
from __future__ import annotations

from typing import Dict, List


def load_ground_truth(app_name: str, rubric_items: List[str], xlsx_path: str, sheet_name: str = "Expert Analysis") -> Dict:
    """Notebook load_ground_truth, unchanged logic (path/sheet now parameters)."""
    import pandas as pd
    df_gt = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    ground_truth = {}
    for item in rubric_items:
        keyword = item.split(". ", 1)[1].lower()
        matched = False
        for length in [20, 15, 10, 7]:
            col_match = [c for c in df_gt.columns if keyword[:length].lower() in c.lower()]
            if col_match:
                ground_truth[item] = int(df_gt.loc[app_name, col_match[0]])
                matched = True
                break
        if not matched:
            words = keyword.split()[:3]
            col_match = [c for c in df_gt.columns if all(w in c.lower() for w in words)]
            ground_truth[item] = int(df_gt.loc[app_name, col_match[0]]) if col_match else 0
    return ground_truth


def evaluate(predictions: Dict, model_name: str, approach: str, app_name: str, ground_truth: Dict,
             rubric_items: List[str]) -> Dict:
    """Notebook evaluate(), unchanged metrics."""
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, precision_score, recall_score)
    num_rubric_items = len(rubric_items)
    y_true, y_pred = [], []
    for item in rubric_items:
        if predictions.get(item) is not None:
            y_true.append(ground_truth[item])
            y_pred.append(predictions[item])
    if not y_true:
        return {}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "model": model_name, "approach": approach, "app": app_name,
        "coverage": len(y_true) / num_rubric_items,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "human_score": sum(ground_truth.values()) / num_rubric_items,
        "llm_score": sum(v for v in predictions.values() if v is not None) / num_rubric_items,
        "predictions": predictions, "y_true": y_true, "y_pred": y_pred,
    }
