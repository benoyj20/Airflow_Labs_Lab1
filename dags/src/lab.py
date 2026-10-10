import json
import os
import shutil
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET = "target"
EXPECTED_FEATURES = 30
MIN_ROWS = 100
RANDOM_STATE = 42

# n_jobs=1 so parallel tasks don't compete for cores
MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=RANDOM_STATE),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    "svm": lambda: SVC(probability=True, random_state=RANDOM_STATE),
}


def load_data(output_dir: str) -> str:
    df = load_breast_cancer(as_frame=True).frame
    path = os.path.join(output_dir, "raw.csv")
    df.to_csv(path, index=False)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns -> {path}")
    return path


def validate_data(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    problems = []

    if TARGET not in df.columns:
        problems.append(f"missing target column '{TARGET}'")
    feature_count = df.shape[1] - (1 if TARGET in df.columns else 0)
    if feature_count != EXPECTED_FEATURES:
        problems.append(f"expected {EXPECTED_FEATURES} feature columns, found {feature_count}")
    if len(df) < MIN_ROWS:
        problems.append(f"expected at least {MIN_ROWS} rows, found {len(df)}")
    missing = int(df.isna().sum().sum())
    if missing:
        problems.append(f"found {missing} missing values")
    if TARGET in df.columns and df[TARGET].nunique() != 2:
        problems.append(f"target must have 2 classes, found {df[TARGET].nunique()}")

    if problems:
        raise ValueError("; ".join(problems))

    class_counts = {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    return {"rows": len(df), "features": feature_count, "class_counts": class_counts}


def split_data(csv_path: str, output_dir: str, test_size: float) -> dict:
    df = pd.read_csv(csv_path)
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    paths = {
        "train": os.path.join(output_dir, "train.csv"),
        "test": os.path.join(output_dir, "test.csv"),
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    print(f"Split into {len(train)} train rows and {len(test)} test rows")
    return paths


def train_and_evaluate(model_name: str, split: dict, model_dir: str) -> dict:
    train = pd.read_csv(split["train"])
    test = pd.read_csv(split["test"])
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    model = Pipeline([("scaler", StandardScaler()), ("classifier", MODELS[model_name]())])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    result = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "model_path": model_path,
    }
    print(f"{model_name}: accuracy={result['accuracy']} f1={result['f1']} roc_auc={result['roc_auc']}")
    return result


def select_best_model(results: list, output_dir: str) -> dict:
    # rank by accuracy, ties broken by roc_auc
    leaderboard = sorted(results, key=lambda r: (r["accuracy"], r["roc_auc"]), reverse=True)

    with open(os.path.join(output_dir, "leaderboard.json"), "w") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"{'rank':<5}{'model':<22}{'accuracy':>9}{'f1':>8}{'roc_auc':>9}")
    for rank, r in enumerate(leaderboard, start=1):
        print(f"{rank:<5}{r['model_name']:<22}{r['accuracy']:>9}{r['f1']:>8}{r['roc_auc']:>9}")
    return leaderboard[0]


def register_model(best: dict, registry_dir: str, run_id: str) -> dict:
    os.makedirs(registry_dir, exist_ok=True)
    registered_path = os.path.join(registry_dir, "best_model.joblib")
    shutil.copyfile(best["model_path"], registered_path)

    metadata = {
        **{k: v for k, v in best.items() if k != "model_path"},
        "registered_path": registered_path,
        "run_id": run_id,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(registry_dir, "best_model.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(registry_dir, "history.jsonl"), "a") as f:
        f.write(json.dumps({**metadata, "decision": "registered"}) + "\n")

    print(f"Registered {best['model_name']} (accuracy={best['accuracy']}) -> {registered_path}")
    return metadata


def reject_model(best: dict, threshold: float, registry_dir: str, run_id: str) -> dict:
    os.makedirs(registry_dir, exist_ok=True)
    record = {
        "model_name": best["model_name"],
        "accuracy": best["accuracy"],
        "threshold": threshold,
        "run_id": run_id,
        "decision": "rejected",
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(registry_dir, "history.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"WARNING: best model {best['model_name']} scored {best['accuracy']}, "
        f"below the threshold of {threshold}. Not registered."
    )
    return record


def predict_sample(model_path: str, test_path: str, n: int = 5) -> list:
    model = joblib.load(model_path)
    test = pd.read_csv(test_path).head(n)
    predictions = model.predict(test.drop(columns=TARGET))
    labels = load_breast_cancer().target_names

    rows = []
    for actual, predicted in zip(test[TARGET], predictions):
        rows.append({"actual": str(labels[int(actual)]), "predicted": str(labels[int(predicted)])})
        print(f"actual={rows[-1]['actual']:<10} predicted={rows[-1]['predicted']}")
    return rows
