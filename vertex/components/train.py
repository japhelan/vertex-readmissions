from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    packages_to_install=["pandas", "scikit-learn", "joblib", "xgboost"],
    base_image="python:3.10-slim",
)
def train_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    train_metrics: Output[Metrics],
    model_type: str = "xgboost",
    hyperparams_json: str = "{}",
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    import json
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, f1_score
    from xgboost import XGBClassifier

    df = pd.read_csv(train_dataset.path, keep_default_na=False, na_values=[""])
    X = df.drop(columns=[id_col, target_col])
    y = df[target_col]

    hyperparams = json.loads(hyperparams_json)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, **hyperparams)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**hyperparams)
    elif model_type == "xgboost":
        model = XGBClassifier(eval_metric="logloss", **hyperparams)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X, y)

    train_preds = model.predict(X)
    train_proba = model.predict_proba(X)[:, 1]

    train_metrics.log_metric("train_roc_auc", float(roc_auc_score(y, train_proba)))
    train_metrics.log_metric("train_f1", float(f1_score(y, train_preds)))
    train_metrics.log_metric("model_type", model_type)

    joblib.dump(model, model_artifact.path)
    print(f"Model trained and saved: {model_type}")
