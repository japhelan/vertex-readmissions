from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    packages_to_install=["pandas", "scikit-learn", "joblib", "xgboost"],
    base_image="python:3.10-slim",
)
def evaluate_model(
    val_dataset: Input[Dataset],
    model_artifact: Input[Model],
    eval_metrics: Output[Metrics],
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    import joblib
    import pandas as pd
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    df = pd.read_csv(val_dataset.path, keep_default_na=False, na_values=[""])
    X = df.drop(columns=[id_col, target_col])
    y = df[target_col]

    model = joblib.load(model_artifact.path)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    f1 = f1_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)

    eval_metrics.log_metric("val_roc_auc", float(roc_auc))
    eval_metrics.log_metric("val_pr_auc", float(pr_auc))
    eval_metrics.log_metric("val_f1", float(f1))
    eval_metrics.log_metric("val_precision", float(precision))
    eval_metrics.log_metric("val_recall", float(recall))

    print(
        f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}"
    )
