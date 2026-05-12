from kfp.v2.dsl import component, Input, Output, Dataset, Metrics


@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.10-slim",
)
def split_data(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    validation_dataset: Output[Dataset],
    split_metrics: Output[Metrics],
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv(input_dataset.path)

    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    ids = df[id_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train, ids.loc[X_train.index]], axis=1)
    val_df = pd.concat([X_val, y_val, ids.loc[X_val.index]], axis=1)

    train_df.to_csv(train_dataset.path, index=False)
    val_df.to_csv(validation_dataset.path, index=False)

    split_metrics.log_metric("train_size", len(train_df))
    split_metrics.log_metric("val_size", len(val_df))
    split_metrics.log_metric("train_positive_rate", float(train_df[target_col].mean()))
    split_metrics.log_metric("val_positive_rate", float(val_df[target_col].mean()))

    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    print("Train target distribution:")
    print(train_df[target_col].value_counts(normalize=True))
    print("Val target distribution:")
    print(val_df[target_col].value_counts(normalize=True))
