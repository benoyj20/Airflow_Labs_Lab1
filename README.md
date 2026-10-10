# Airflow Lab 1 - Model Selection Pipeline

For this lab I took the original Airflow Lab 1 (a KMeans clustering DAG) and rebuilt the DAG into a
model selection pipeline. It trains four classifiers on the Breast Cancer Wisconsin dataset at the
same time, picks the best one, and only saves it to a "model registry" folder if its accuracy is
above a threshold. Everything runs in Docker, so you don't need Python or Airflow installed locally.

```
load_data -> validate_data -> split_data -> train_model (x4, in parallel) -> pick_best -> quality_gate

quality_gate passes -> register_model -> predict_sample -> pipeline_summary
quality_gate fails  -> reject_model ---------------------> pipeline_summary
```

## What I changed

The original DAG was four tasks in a straight line: load the credit card CSV, preprocess it, fit
KMeans for k = 1 to 49 inside a single task, then use the elbow method to pick k. It worked, but the
whole model-fitting step lived in one task, so Airflow wasn't really orchestrating much. THE run took over 9 minutes to finish.

These are the changes I made to the DAG:

- **Parallel training with dynamic task mapping.** `train_model` is defined once and expanded with
  `.expand(model_name=[...])`, so Airflow creates one task per model (logistic regression, random
  forest, gradient boosting, SVM) and runs them at the same time. `pick_best` then waits for all four
  and gets their results as a list. I also set `map_index_template` so the UI shows
  `train_model [svm]` instead of `train_model [3]`.
- **Branching.** `quality_gate` is a `@task.branch` task. If the best model's accuracy is at least
  the threshold it goes to `register_model`, otherwise it goes to `reject_model`. The path that
  isn't taken gets marked as skipped.
- **Trigger rule on the last task.** `pipeline_summary` comes after both branches, and one of them is
  always skipped, so with the default rule it would never run. I set it to
  `none_failed_min_one_success` so it runs no matter which path was taken.
- **Data validation.** `validate_data` checks the column count, row count, missing values and that
  there are two classes. If something is wrong it raises `AirflowFailException`, which fails the
  task right away instead of retrying (bad data won't fix itself on retry).
- **Params.** `accuracy_threshold` (default 0.95) and `test_size` (default 0.2) are DAG params, so
  you can change them from the Trigger form without editing code.
- **Passing file paths through XCom instead of data.** The original pickled the whole DataFrame and
  pushed it through XCom (which lives in Airflow's database). Now the tasks write CSVs and models to
  `working_data/` and only pass file paths and scores between tasks.
- **Run settings.** `max_active_runs=1` so extra triggers wait in a queue instead of running on top
  of each other, `retries=2` with a 1 minute delay, a 10 minute
  timeout per task, a 30 minute timeout per run, and an `on_failure_callback` that logs which task
  failed.
- **Model registry.** The winning model is copied to `working_data/model_registry/best_model.joblib`
  along with a JSON file of its scores, and every run adds a line to `history.jsonl` saying whether
  the model was registered or rejected. This also fixes something odd in the original, which saved
  the k=49 model even though the elbow method picked a different k.

Other changes:

- Switched the dataset to Breast Cancer Wisconsin (it ships with scikit-learn, so there's nothing to
  download) and the model to a comparison of four classifiers.
- Split the code into `dags/model_selection_dag.py` (the DAG) and `dags/src/lab.py` (the ML
  functions, with no Airflow code in them). I also stopped calling the DAG file `airflow.py`, since
  that name clashes with the `airflow` package.
- Pinned `pandas==2.1.4` and `scikit-learn==1.9.1` in `docker-compose.yaml` and removed `kneed`,
  which isn't used anymore.

With these changes a full run takes about 25 seconds.

## What you need to install

You only need two things:

1. **Docker Desktop**, which includes Docker Compose. Download it from
   https://www.docker.com/products/docker-desktop/. I used Docker Engine 29.8 with Compose v5.5, but
   any recent version with Compose v2 (`docker compose`, not `docker-compose`) should work.
2. **Git**, from https://git-scm.com/downloads.

Your machine should have at least 4 GB of RAM available to Docker (8 GB is better), around 5 GB of
free disk space, and nothing else running on port 8080.

Open Docker Desktop and wait until it says the engine is running, then check both are installed:

```bash
docker --version
docker compose version
git --version
```

You don't need to install Python or any libraries. These all run inside the containers and get
downloaded automatically:

- Apache Airflow 2.9.2 (`apache/airflow:2.9.2` image, which uses Python 3.12)
- pandas 2.1.4 and scikit-learn 1.9.1
- PostgreSQL 13 (Airflow's metadata database)
- Redis 7.2 (the queue between the scheduler and the worker)

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/benoyj20/Airflow_Labs_Lab1.git
cd Airflow_Labs_Lab1
```

**2. Create the folders Airflow mounts**

These aren't in the repo because they're empty or only hold generated files.

On macOS, Linux or Git Bash:
```bash
mkdir -p logs plugins config working_data
```

On Windows PowerShell:
```powershell
mkdir logs, plugins, config, working_data
```

**3. Create the `.env` file**

This sets the user ID the containers run as, so the files they write belong to you.

On macOS or Linux:
```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

On Windows PowerShell (Windows has no `id -u`, so use Airflow's default of 50000):
```powershell
"AIRFLOW_UID=50000" | Out-File -Encoding ascii .env
```

**4. Initialize Airflow**

You only need to do this the first time.

```bash
docker compose up airflow-init
```

The first time, it has to download the images, which takes a few minutes. It's done when you see
something like:

```
airflow-init-1  | User "airflow2" created with role "Admin"
airflow-init-1  | 2.9.2
airflow-init-1 exited with code 0
```

"exited with code 0" means it worked. It set up the database and created the login user.

**5. Start Airflow**

```bash
docker compose up -d
```

The `-d` runs it in the background. Every time the containers start they install pandas and
scikit-learn, so give it 1 to 3 minutes. Run `docker compose ps` until every service says
`(healthy)`:

```
SERVICE             STATUS
airflow-scheduler   Up 2 minutes (healthy)
airflow-triggerer   Up 2 minutes (healthy)
airflow-webserver   Up 2 minutes (healthy)
airflow-worker      Up 2 minutes (healthy)
postgres            Up 2 minutes (healthy)
redis               Up 2 minutes (healthy)
```

**6. Log in**

Go to http://localhost:8080 and log in with username `airflow2` and password `airflow2`.

## Running the pipeline

The DAG is called `Airflow_Lab1_model_selection`. Since the example DAGs are turned off, it should be
the only one on the DAGs page.

1. Turn on the toggle next to the DAG name. New DAGs start paused, and because this one has a daily
   schedule, turning it on starts a run right away.
2. Click the DAG name and open the Grid view to watch it. It should finish in about 25 seconds.

To run it again, click the play button (Trigger DAG) in the top right. Because the DAG has params,
Airflow shows a form with `accuracy_threshold` and `test_size`. Leave the defaults and click
Trigger. If you trigger it several times, the extra runs show as queued and run one after another.

**To see the reject branch:** trigger it again, but this time set `accuracy_threshold` to `0.999` in
the form. No model gets that high, so `quality_gate` sends the run to `reject_model`, and
`register_model` and `predict_sample` show up as skipped.

## What a successful run looks like

In the Grid view, every task should be green except `reject_model`, which is pink (skipped) on a
normal run. The four `train_model` tasks should all have the same start time, which shows they ran
in parallel. Each one takes 1 to 2 seconds, and everything else takes under a second.

The train/test split uses a fixed seed (42), so you should get exactly these scores:

| Model | Accuracy | F1 | ROC AUC |
|---|---|---|---|
| logistic_regression | 0.9825 | 0.9861 | 0.9954 |
| svm | 0.9825 | 0.9861 | 0.9950 |
| random_forest | 0.9561 | 0.9655 | 0.9932 |
| gradient_boosting | 0.9561 | 0.9660 | 0.9907 |

Logistic regression and SVM tie on accuracy, and logistic regression wins on ROC AUC, so it's the one
that gets registered.

You can check this in the task logs (click a task square, then Logs):

- `pick_best` prints the table above.
- `quality_gate` prints `Quality gate: logistic_regression accuracy=0.9825 threshold=0.95 -> PASS`.
- `predict_sample` loads the saved model back from disk and predicts 5 test rows, printing lines
  like `actual=malignant  predicted=malignant`.
- `pipeline_summary` prints
  `Run finished: best model=logistic_regression accuracy=0.9825 decision=REGISTERED` and
  `Smoke test: 5/5 sample predictions correct`.

The run also writes these files into `working_data/` on your machine:

```
working_data/
  model_registry/
    best_model.joblib     the registered model (scaler + classifier)
    best_model.json       its scores, which run it came from, and when
    history.jsonl         one line per run, registered or rejected
  runs/<run_id>/
    raw.csv, train.csv, test.csv
    leaderboard.json      all four models ranked
    models/               every candidate model
```

## Stopping Airflow

```bash
docker compose down
```

This removes the containers but keeps the database, so your run history and login are still there
next time. Start it again with `docker compose up -d`; you don't need `airflow-init` again.

If you want to wipe everything and start fresh, use `docker compose down -v`, then go back to step 4.

## If something goes wrong

- **"Cannot connect to the Docker daemon"**: Docker Desktop isn't running. Start it and try again.
- **localhost:8080 won't load**: the containers are probably still installing packages. Wait until
  `docker compose ps` shows everything as healthy.
- **"port is already allocated"**: something else is using port 8080. Either stop it, or change
  `"8080:8080"` to `"8081:8080"` in `docker-compose.yaml` and go to http://localhost:8081.
- **The DAG doesn't show up**: run
  `docker compose exec airflow-scheduler airflow dags list-import-errors` to see what's wrong.
- **Permission denied under `working_data` (Linux)**: Docker created the folder as root because it
  didn't exist yet. Make sure you did step 2 before starting, or run
  `sudo chown -R $(id -u) working_data`.
- **`curl -o` doesn't work in Windows PowerShell**: PowerShell's `curl` is a different command. Use
  `curl.exe`.

## Files

```
dags/
  model_selection_dag.py   the DAG
  src/
    __init__.py
    lab.py                 ML functions: load, validate, split, train, select, register, predict
docker-compose.yaml        the official Airflow 2.9.2 compose file, with the changes below
.gitignore
README.md
```

Changes I made to the official `docker-compose.yaml`:

- `AIRFLOW__CORE__LOAD_EXAMPLES: 'false'` so the example DAGs don't show up
- `_PIP_ADDITIONAL_REQUIREMENTS` set to `pandas==2.1.4 scikit-learn==1.9.1`
- added a `./working_data:/opt/airflow/working_data` volume for the pipeline output
- changed the admin login to `airflow2` / `airflow2`
