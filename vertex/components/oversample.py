from kfp.dsl import component, Input, Output, Dataset


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn", "imbalanced-learn"],
    base_image="python:3.10-slim",
)
def oversample_training(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
    random_state: int = 42,
):
    import pandas as pd
    from imblearn.over_sampling import RandomOverSampler

    df = pd.read_csv(input_dataset.path, keep_default_na=False, na_values=[""])

    print(f"Input shape: {df.shape}")
    print("Class distribution BEFORE oversampling:")
    print(df[target_col].value_counts(dropna=False))

    missing = [c for c in [target_col, id_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    resampled_df = X_resampled.copy()
    resampled_df[target_col] = y_resampled

    print("Class distribution AFTER oversampling:")
    print(resampled_df[target_col].value_counts(dropna=False))
    print(f"Output shape: {resampled_df.shape}")

    resampled_df.to_csv(output_dataset.path, index=False)
