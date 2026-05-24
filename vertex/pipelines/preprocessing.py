from kfp.dsl import pipeline

from vertex.components.ingest import load_validate_data
from vertex.components.split import split_data
from vertex.components.oversample import oversample_training
from vertex.components.preprocessing import (
    fit_apply_preprocessing_v1,
    apply_preprocessing_v1,
)


@pipeline(name="readmissions-preprocessing-pipeline")
def preprocessing_pipeline(
    dataset_gcs_uri: str,
    dataset_version: str = "v0.0",
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    validated = (
        load_validate_data(  # pyright: ignore[reportCallIssue]
            dataset_gcs_uri=dataset_gcs_uri,
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )

    split = (
        split_data(  # pyright: ignore[reportCallIssue]
            input_dataset=validated.outputs["output_dataset"],
            test_size=test_size,
            random_state=random_state,
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )

    oversampled = (
        oversample_training(  # pyright: ignore[reportCallIssue]
            input_dataset=split.outputs["train_dataset"],
            random_state=random_state,
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )

    preprocessed_train = (
        fit_apply_preprocessing_v1(  # pyright: ignore[reportCallIssue]
            input_dataset=oversampled.outputs["output_dataset"],
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("2")
        .set_memory_limit("8G")
    )

    (
        apply_preprocessing_v1(  # pyright: ignore[reportCallIssue]
            input_dataset=split.outputs["validation_dataset"],
            preprocessing_artifacts=preprocessed_train.outputs[
                "preprocessing_artifacts"
            ],
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )
