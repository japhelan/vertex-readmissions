from kfp.v2.dsl import component, Input, Output, Dataset, Artifact, Metrics


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
    base_image="python:3.10-slim",
)
def fit_apply_preprocessing_v1(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    preprocessing_artifacts: Output[Artifact],
    preprocessing_metrics: Output[Metrics],
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    import os
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["age_group"] = pd.cut(
            X["age"],
            bins=[0, 18, 25, 40, 65, 80, np.inf],
            labels=["0-18", "19-25", "26-40", "41-65", "66-80", "81+"],
            right=False,
        ).astype(str)
        X = X.drop(columns=["age"])
        X["medications_prescribed"] = (
            X["medications_prescribed"].replace("", pd.NA).astype(float).apply(lambda x: 1 if x > 0 else 0)
        )
        X["number_of_prior_visits"] = X["number_of_prior_visits"].replace("", pd.NA).astype(float)
        X["length_of_stay_score"] = X["length_of_stay"].apply(
            lambda x: 1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else (4 if x <= 6 else (5 if x <= 14 else 7))))
        )
        X = X.drop(columns=["length_of_stay"])
        return X

    df = pd.read_csv(input_dataset.path)
    print(f"Input training shape: {df.shape}")

    ids = df[[id_col]].copy()
    y = df[[target_col]].copy()
    X = df.drop(columns=[id_col, target_col]).copy()

    X = engineer_features(X)

    num_cols = ["height_m", "bmi", "adjusted_weight_kg", "number_of_prior_visits", "length_of_stay_score"]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ],
        remainder="passthrough",
    )

    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    prepared_df = pd.concat(
        [ids.reset_index(drop=True), X_df, y.reset_index(drop=True)], axis=1
    )
    print(f"Output training shape: {prepared_df.shape}")
    prepared_df.to_csv(output_dataset.path, index=False)

    os.makedirs(preprocessing_artifacts.path, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(preprocessing_artifacts.path, "preprocessor.joblib"))
    print("Preprocessing artifact saved:", os.listdir(preprocessing_artifacts.path))

    preprocessing_metrics.log_metric("input_features", int(X.shape[1]))
    preprocessing_metrics.log_metric("output_features", int(X_df.shape[1]))
    preprocessing_metrics.log_metric("train_samples", int(len(prepared_df)))


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
    base_image="python:3.10-slim",
)
def apply_preprocessing_v1(
    input_dataset: Input[Dataset],
    preprocessing_artifacts: Input[Artifact],
    output_dataset: Output[Dataset],
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    import os
    import joblib
    import numpy as np
    import pandas as pd

    def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["age_group"] = pd.cut(
            X["age"],
            bins=[0, 18, 25, 40, 65, 80, np.inf],
            labels=["0-18", "19-25", "26-40", "41-65", "66-80", "81+"],
            right=False,
        ).astype(str)
        X = X.drop(columns=["age"])
        X["medications_prescribed"] = (
            X["medications_prescribed"].replace("", pd.NA).astype(float).apply(lambda x: 1 if x > 0 else 0)
        )
        X["number_of_prior_visits"] = X["number_of_prior_visits"].replace("", pd.NA).astype(float)
        X["length_of_stay_score"] = X["length_of_stay"].apply(
            lambda x: 1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else (4 if x <= 6 else (5 if x <= 14 else 7))))
        )
        X = X.drop(columns=["length_of_stay"])
        return X

    df = pd.read_csv(input_dataset.path)
    print(f"Input validation shape: {df.shape}")

    preprocessor_path = os.path.join(preprocessing_artifacts.path, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise ValueError(f"Missing preprocessor artifact: {preprocessor_path}")

    preprocessor = joblib.load(preprocessor_path)

    ids = df[[id_col]].copy()
    y = df[[target_col]].copy()
    X = df.drop(columns=[id_col, target_col]).copy()

    X = engineer_features(X)

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    prepared_df = pd.concat(
        [ids.reset_index(drop=True), X_df, y.reset_index(drop=True)], axis=1
    )
    print(f"Output validation shape: {prepared_df.shape}")
    prepared_df.to_csv(output_dataset.path, index=False)
