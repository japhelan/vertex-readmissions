from kfp.v2.dsl import component, Input, Output, Dataset


@component(
    packages_to_install=["pandas", "numpy", "fsspec", "gcsfs"],
    base_image="python:3.10-slim",
)
def load_validate_data(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    import pandas as pd
    import numpy as np

    def format_columns(df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()

        if "PatientID" in df_transformed.columns:
            df_transformed = df_transformed.rename(columns={"PatientID": "patient_id"})

        df_transformed.columns = (
            df_transformed.columns.str.strip().str.lower().str.replace(" ", "_")
        )
        df_transformed.columns = df_transformed.columns.str.replace("(", "").str.replace(")", "")
        return df_transformed

    df = pd.read_csv(input_dataset.uri)

    if df.empty:
        raise ValueError("Input dataset is empty")

    print(f"Dataset shape: {df.shape}")

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df = format_columns(df)

    missing = [c for c in [target_col, id_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Target distribution:")
    print(df[target_col].value_counts(dropna=False))

    if df[target_col].dropna().nunique() < 2:
        raise ValueError(f"Target column '{target_col}' must contain at least two classes")

    df.to_csv(output_dataset.path, index=False)

    output_dataset.metadata["source_path"] = input_dataset.uri
    output_dataset.metadata["row_count"] = len(df)
