from kfp.v2 import dsl
from kfp.v2.dsl import Dataset
from kfp.dsl import pipeline

from vertex.components.ingest import load_validate_data
from vertex.components.split import split_data
from vertex.components.oversample import oversample_training
from vertex.components.preprocessing import (
    fit_apply_preprocessing_v1,
    apply_preprocessing_v1,
)
from vertex.components.train import train_model
from vertex.components.evaluate import evaluate_model


@pipeline(name="readmissions-training-pipeline")
def training_pipeline(
    training_dataset_path: str,
    dataset_version: str = "v0.0",
    dataset_resource_name: str = "",
    model_type: str = "xgboost",
    hyperparams_json: str = "{}",
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = "readmission_within_30_days",
    id_col: str = "patient_id",
):
    imported_dataset = dsl.importer(
        artifact_uri=training_dataset_path,
        artifact_class=Dataset,
        reimport=False,
        metadata={"resourceName": dataset_resource_name},
    )

    validated = (
        load_validate_data(
            input_dataset=imported_dataset.output,
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )

    split = (
        split_data(
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
        oversample_training(
            input_dataset=split.outputs["train_dataset"],
            random_state=random_state,
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )

    preprocessed_train = (
        fit_apply_preprocessing_v1(
            input_dataset=oversampled.outputs["output_dataset"],
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("2")
        .set_memory_limit("8G")
    )

    preprocessed_val = (
        apply_preprocessing_v1(
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

    trained = (
        train_model(
            train_dataset=preprocessed_train.outputs["output_dataset"],
            model_type=model_type,
            hyperparams_json=hyperparams_json,
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("2")
        .set_memory_limit("8G")
    )

    (
        evaluate_model(
            val_dataset=preprocessed_val.outputs["output_dataset"],
            model_artifact=trained.outputs["model_artifact"],
            target_col=target_col,
            id_col=id_col,
        )
        .set_cpu_limit("1")
        .set_memory_limit("4G")
    )
