import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import aiplatform, storage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(
    VERSION_ID: str,
    DATASET_GCS_URI: str,
    DATASET_LOCAL_PATH: Path,
    LOCAL_VERSION_DIR: Path,
    description: str = "",
    tags: list[str] | None = None,
):
    dataset_df = pd.read_csv(DATASET_LOCAL_PATH)
    manifest = {
        "version_id": VERSION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_uri": DATASET_GCS_URI,
        "source_file": str(DATASET_LOCAL_PATH.resolve()),
        "sha256": sha256_file(DATASET_LOCAL_PATH),
        "row_count": int(dataset_df.shape[0]),
        "column_count": int(dataset_df.shape[1]),
        "columns": [str(c) for c in dataset_df.columns.tolist()],
        "label_column": "target",
        "description": description,
        "tags": tags or [],
    }

    manifest_path = LOCAL_VERSION_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, manifest


def log_dataset_to_gcs(
    DATASET_LOCAL_PATH: Path,
    VERSION_ID: str,
    BUCKET_ROOT_URI: str,
    PROJECT_ID: str,
    LOCATION: str,
    log_experiment: bool,
    EXPERIMENT_NAME: str,
    description: str = "",
    tags: list[str] | None = None,
    resume_run: bool = False,
):
    """
    Log a dataset version to GCS and optionally record metadata in Vertex Experiments.

    Args:
        description: Human-readable summary of what changed in this version.
        tags:        List of labels e.g. ["cleaned", "outliers-removed", "v0"].
        resume_run:  If True, resume an existing run with the same name.
    """
    DATASET_GCS_URI = f"{BUCKET_ROOT_URI}/datasets/readmissions/{VERSION_ID}/train.csv"
    LOCAL_VERSION_DIR = Path("../data/versions") / VERSION_ID
    LOCAL_VERSION_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path, manifest = make_manifest(
        VERSION_ID,
        DATASET_GCS_URI,
        DATASET_LOCAL_PATH,
        LOCAL_VERSION_DIR,
        description=description,
        tags=tags,
    )

    client = storage.Client(project=PROJECT_ID)
    bucket_name = BUCKET_ROOT_URI.replace("gs://", "").split("/")[0]
    bucket = client.bucket(bucket_name)

    dataset_blob_path = f"datasets/readmissions/{VERSION_ID}/train.csv"
    manifest_blob_path = f"datasets/readmissions/{VERSION_ID}/manifest.json"

    bucket.blob(dataset_blob_path).upload_from_filename(str(DATASET_LOCAL_PATH))
    bucket.blob(manifest_blob_path).upload_from_filename(str(manifest_path))

    print(f"Uploaded: gs://{bucket_name}/{dataset_blob_path}")
    print(f"Uploaded: gs://{bucket_name}/{manifest_blob_path}")

    if log_experiment:
        raw_run_name = f"readmissions-data-{VERSION_ID}"
        RUN_NAME = re.sub(r"[^a-z0-9-]+", "-", raw_run_name.lower()).strip("-")[:120]

        aiplatform.init(
            project=PROJECT_ID, location=LOCATION, experiment=EXPERIMENT_NAME
        )
        aiplatform.start_run(run=RUN_NAME, resume=resume_run)
        aiplatform.log_params(
            {
                "dataset_version": VERSION_ID,
                "dataset_uri": DATASET_GCS_URI,
                "manifest_uri": f"{BUCKET_ROOT_URI}/datasets/readmissions/{VERSION_ID}/manifest.json",
                "row_count": manifest["row_count"],
                "column_count": manifest["column_count"],
                "description": description,
                "tags": ",".join(tags or []),
            }
        )
        aiplatform.end_run()
        print(
            f"Logged dataset version to Vertex Experiments: {EXPERIMENT_NAME} / {RUN_NAME}"
        )


def _extract_task_metrics(pipeline_job, task_name: str = "evaluate-model") -> dict:
    """Pull logged metrics out of a completed pipeline task's Metrics artifact."""
    for task in pipeline_job.task_details:
        if task.task_name == task_name:
            artifact_list = task.outputs.get("metrics")
            if artifact_list and artifact_list.artifacts:
                return dict(artifact_list.artifacts[0].metadata)
    return {}


def log_pipeline_run(
    pipeline_job,
    dataset_version: str,
    training_dataset_path: str,
    model_version: str,
    PROJECT_ID: str,
    LOCATION: str,
    EXPERIMENT_NAME: str = "readmissions-pipeline-runs",
    wait_for_completion: bool = False,
    eval_task_name: str = "evaluate-model",
):
    """
    Record which dataset version and model version a pipeline job was run against.
    Optionally waits for the job to finish and logs eval metrics from the pipeline.

    Args:
        pipeline_job:           Submitted aiplatform.PipelineJob instance.
        dataset_version:        Dataset version tag used as input (e.g. "v1.3").
        training_dataset_path:  Full GCS path to the training CSV.
        model_version:          Model version label for this run (e.g. "v0").
        EXPERIMENT_NAME:        Vertex Experiment to log into.
        wait_for_completion:    Block until the pipeline finishes and log eval metrics.
        eval_task_name:         KFP task name that produces the Metrics artifact.
    """
    run_name = re.sub(
        r"[^a-z0-9-]+",
        "-",
        f"pipeline-run-{model_version}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    ).strip("-")[:120]

    aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment=EXPERIMENT_NAME)
    aiplatform.start_run(run=run_name)
    aiplatform.log_params(
        {
            "dataset_version": dataset_version,
            "training_dataset_path": training_dataset_path,
            "model_version": model_version,
            "pipeline_job_display_name": pipeline_job.display_name,
            "pipeline_resource_name": pipeline_job.resource_name,
        }
    )

    if wait_for_completion:
        print("Waiting for pipeline to complete...")
        pipeline_job.wait()
        metrics = _extract_task_metrics(pipeline_job, task_name=eval_task_name)
        if metrics:
            aiplatform.log_metrics(metrics)
            print(f"Logged eval metrics: {metrics}")
        else:
            print(f"Warning: no metrics found in task '{eval_task_name}'")

    aiplatform.end_run()
    print(f"Logged pipeline run to Vertex Experiments: {EXPERIMENT_NAME} / {run_name}")
