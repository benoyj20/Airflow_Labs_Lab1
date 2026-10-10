import logging
import os
import re
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.utils.trigger_rule import TriggerRule

from src import lab

WORKING_DIR = os.environ.get("PIPELINE_WORKING_DIR", "/opt/airflow/working_data")
REGISTRY_DIR = os.path.join(WORKING_DIR, "model_registry")

log = logging.getLogger(__name__)


def _run_dir() -> str:
    # run ids contain ':' and '+', which aren't allowed in Windows folder names
    run_id = get_current_context()["run_id"]
    path = os.path.join(WORKING_DIR, "runs", re.sub(r"[^A-Za-z0-9_.-]", "_", run_id))
    os.makedirs(path, exist_ok=True)
    return path


def log_task_failure(context):
    ti = context["task_instance"]
    log.error(
        "PIPELINE FAILURE: task=%s run=%s try=%s error=%s",
        ti.task_id, context["run_id"], ti.try_number, context.get("exception"),
    )


default_args = {
    "owner": "benoyj20",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=10),
    "on_failure_callback": log_task_failure,
}


@dag(
    dag_id="Airflow_Lab1_model_selection",
    description="Parallel model training with a quality gate on the Breast Cancer dataset",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    default_args=default_args,
    params={
        "accuracy_threshold": Param(
            0.95, type="number", minimum=0, maximum=1,
            description="Minimum test accuracy the best model needs to be registered",
        ),
        "test_size": Param(
            0.2, type="number", minimum=0.1, maximum=0.5,
            description="Fraction of rows held out for testing",
        ),
    },
    tags=["lab1", "model-selection"],
)
def model_selection_pipeline():

    @task
    def load_data() -> str:
        return lab.load_data(_run_dir())

    # no point retrying bad data
    @task(retries=0)
    def validate_data(csv_path: str) -> str:
        try:
            report = lab.validate_data(csv_path)
        except ValueError as err:
            raise AirflowFailException(f"Data validation failed: {err}") from err
        log.info("Data validation passed: %s", report)
        return csv_path

    @task
    def split_data(csv_path: str, **context) -> dict:
        return lab.split_data(csv_path, _run_dir(), test_size=float(context["params"]["test_size"]))

    @task(map_index_template="{{ model_name }}")
    def train_model(model_name: str, split: dict) -> dict:
        get_current_context()["model_name"] = model_name
        return lab.train_and_evaluate(model_name, split, os.path.join(_run_dir(), "models"))

    @task
    def pick_best(results: list) -> dict:
        return lab.select_best_model(list(results), _run_dir())

    @task.branch
    def quality_gate(best: dict, **context) -> str:
        threshold = float(context["params"]["accuracy_threshold"])
        passed = best["accuracy"] >= threshold
        log.info(
            "Quality gate: %s accuracy=%s threshold=%s -> %s",
            best["model_name"], best["accuracy"], threshold, "PASS" if passed else "FAIL",
        )
        return "register_model" if passed else "reject_model"

    @task
    def register_model(best: dict, **context) -> dict:
        return lab.register_model(best, REGISTRY_DIR, context["run_id"])

    @task
    def predict_sample(registered: dict, split: dict) -> list:
        return lab.predict_sample(registered["registered_path"], split["test"])

    @task
    def reject_model(best: dict, **context) -> dict:
        threshold = float(context["params"]["accuracy_threshold"])
        return lab.reject_model(best, threshold, REGISTRY_DIR, context["run_id"])

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def pipeline_summary(best: dict, predictions=None, rejection=None):
        decision = "REJECTED" if rejection else "REGISTERED"
        log.info("Run finished: best model=%s accuracy=%s decision=%s",
                 best["model_name"], best["accuracy"], decision)
        if predictions:
            correct = sum(p["actual"] == p["predicted"] for p in predictions)
            log.info("Smoke test: %s/%s sample predictions correct", correct, len(predictions))
        return {"best_model": best["model_name"], "accuracy": best["accuracy"], "decision": decision}

    validated = validate_data(load_data())
    split = split_data(validated)
    results = train_model.partial(split=split).expand(model_name=list(lab.MODELS))
    best = pick_best(results)

    gate = quality_gate(best)
    registered = register_model(best)
    predictions = predict_sample(registered, split)
    rejection = reject_model(best)
    gate >> [registered, rejection]

    pipeline_summary(best, predictions, rejection)


model_selection_pipeline()
