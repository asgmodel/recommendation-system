"""One-command, auditable HLO training/evaluation pipeline for Google Colab.

The synthetic mode performs real optimization on demo data, not real-world
learning evidence. Reviewed mode requires independently supplied reviewed data.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

REPOSITORY = "https://github.com/asgmodel/recommendation-system.git"
REQUIRED_SPLITS = ("train", "validation", "test")


def heading(text: str) -> None:
    print(f"\n{'=' * 12} {text} {'=' * 12}", flush=True)


def command(args: list[str], *, cwd: Path, preview: bool = False) -> dict:
    quoted = " ".join(map(str, args))
    print("$", quoted, flush=True)
    if preview:
        return {}
    started = time.monotonic()
    process = subprocess.Popen(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"  ⏳ Still running ({int(time.monotonic() - started)} s)...", flush=True)
    except BaseException:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    if process.returncode:
        print(stdout[-4000:], file=sys.stderr)
        print(stderr[-6000:], file=sys.stderr)
        raise RuntimeError(f"Command failed ({process.returncode}): {quoted}")
    print(f"  ✓ Completed in {time.monotonic() - started:.1f}s", flush=True)
    # The CLI returns JSON; pip and git do not.
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        if stdout.strip():
            print(stdout[-1500:], flush=True)
        return {}


def hlo(root: Path, *args: str | Path, preview: bool = False) -> dict:
    return command(
        [sys.executable, "-m", "hlo_recommender", *map(str, args)],
        cwd=root,
        preview=preview,
    )


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_dataset(root: Path, mode: str, require_policy: bool) -> dict:
    """Fail early rather than claim reviewed learning from the synthetic set."""
    base = root / "data" / ("development" if mode == "synthetic" else "reviewed")
    paths = {}
    for prefix in ("questions", "assessments"):
        for split in REQUIRED_SPLITS:
            key = f"{prefix}.{split}"
            paths[key] = base / f"{key}.jsonl"
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing training/evaluation files: " + ", ".join(missing)
            + ". Real training needs reviewed questions and assessments; they are NOT in this GitHub repository."
        )
    rows = {key: jsonl(path) for key, path in paths.items()}
    for key, records in rows.items():
        if not records:
            raise ValueError(f"Dataset is empty: {paths[key]}")
        statuses = {r.get("data_status") for r in records}
        if statuses != {mode}:
            raise ValueError(f"{key}: expected data_status={mode}, got {statuses}")
    if any("relevance" not in r for r in rows["questions.test"]):
        raise ValueError("Ranking evaluation requires relevance labels for every test question")

    policy_files = [base / f"transitions.{split}.jsonl" for split in ("train", "validation")]
    policy_ready = all(path.is_file() for path in policy_files)
    if require_policy and not policy_ready:
        raise FileNotFoundError("Dueling DQN needs reviewed transition files: " + ", ".join(map(str, policy_files)))
    if policy_ready and mode == "synthetic":
        # The checked-in policy config expects real reviewed transitions.
        raise ValueError("Policy training in synthetic mode is not supported by the repository config")

    return {
        "source": str(base),
        "status": mode,
        "counts": {key: len(value) for key, value in rows.items()},
        "questions": rows["questions.test"],
        "policy_ready": policy_ready,
    }


def config_for_run(root: Path, config_name: str, output: Path, run_id: str) -> Path:
    source = root / "configs" / config_name
    text = source.read_text(encoding="utf-8")
    relative_output = output.relative_to(root).as_posix()
    text, count = re.subn(
        r'(?m)^output\s*=\s*"[^"]+"$',
        f'output = "{relative_output}"',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"No output setting in {source}")
    target = root / "configs" / f"full_{source.stem.replace('.', '_')}_{run_id}.toml"
    target.write_text(text, encoding="utf-8")
    return target


def write_statistics(root: Path, mode: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    base = root / "data" / ("development" if mode == "synthetic" else "reviewed")
    stats = []
    for split in REQUIRED_SPLITS:
        qs = jsonl(base / f"questions.{split}.jsonl")
        ev = jsonl(base / f"assessments.{split}.jsonl")
        stats.append({
            "split": split,
            "questions": len(qs),
            "assessments": len(ev),
            "learners": len({r["learner_id"] for r in ev}),
            "correct_rate": sum(int(r["correct"]) for r in ev) / len(ev),
        })
    df = pd.DataFrame(stats)
    df.to_csv(out / "dataset_statistics.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.plot(x="split", y="assessments", kind="bar", ax=axes[0], legend=False, title="Assessment records")
    df.plot(x="split", y="correct_rate", kind="bar", ax=axes[1], legend=False, ylim=(0, 1),
            title="Correct-answer rate")
    for axis in axes:
        axis.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    figure.savefig(out / "dataset_statistics.png", dpi=130)
    plt.close(figure)


def write_evaluation_charts(out: Path, bkt_out: Path, experiment_out: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    with (bkt_out / "metrics.json").open() as fh:
        bkt = json.load(fh)
    calib = pd.DataFrame(bkt["calibration"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.plot(calib["predicted"], calib["observed"], "o-", label="BKT")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted", ylabel="Observed", title="BKT calibration")
    ax.legend()
    fig.savefig(out / "bkt_calibration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    ranking = pd.read_csv(experiment_out / "metrics.csv")
    ranking.set_index("arm")[["mrr", "map_at_k", "ndcg_at_k"]].plot(
        kind="bar", figsize=(9, 4), ylim=(0, 1), title="Recommendation ranking"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "ranking_metrics.png", dpi=140)
    plt.close()


def write_mt5_history_charts(out: Path, mt5_out: Path) -> None:
    """Plot actual training losses and validation metrics saved by the model trainer."""
    import matplotlib.pyplot as plt
    import pandas as pd

    history_path = mt5_out / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError("mT5 training returned no epoch history")
    rows = [
        {
            "epoch": entry["epoch"],
            "train_loss": entry["train_loss"],
            "validation_nonempty_f1": entry["validation"].get("nonempty_f1"),
            "validation_valid_rate": entry["validation"].get("valid_rate"),
        }
        for entry in history
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "mt5_training_history.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.plot(x="epoch", y="train_loss", marker="o", ax=axes[0], title="mT5 training loss")
    df.plot(x="epoch", y="validation_nonempty_f1", marker="o", ax=axes[1],
            title="mT5 validation answer F1", ylim=(0, 1))
    plt.tight_layout()
    figure.savefig(out / "mt5_training_history.png", dpi=140)
    plt.close(figure)


def train_pipeline(
    *,
    mode: str = "synthetic",
    root: str | Path | None = None,
    install: bool = True,
    require_gpu: bool = True,
    require_policy: bool = False,
    dry_run: bool = False,
    save_to_drive: bool = False,
    mt5_batch_size: int = 1,
) -> dict:
    """Fit models, independently test them, and save artifacts with one call."""
    if mode not in ("synthetic", "reviewed"):
        raise ValueError("mode must be 'synthetic' or 'reviewed'")
    if sys.version_info < (3, 11):
        raise RuntimeError("Python >=3.11 is required")
    if type(mt5_batch_size) is not int or mt5_batch_size < 1:
        raise ValueError("mt5_batch_size must be a positive integer")
    if root is None:
        current = Path.cwd()
        if (current / "pyproject.toml").is_file() and (current / "src" / "hlo_recommender").is_dir():
            root = current
        elif Path("/content").is_dir():
            root = Path("/content/recommendation-system")
        else:
            root = current / "recommendation-system"
    root = Path(root)
    root = root.resolve()
    heading("01 Repository")
    if not (root / "pyproject.toml").is_file():
        if dry_run:
            raise FileNotFoundError("For --dry-run, clone the repo first to validate the data")
        root.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--depth", "1", REPOSITORY, str(root)], cwd=root.parent)
    heading("02 Data gate")
    dataset = check_dataset(root, mode, require_policy)
    print(json.dumps({"data_status": mode, "counts": dataset["counts"],
                      "policy_training_available": dataset["policy_ready"]}, indent=2))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    out = root / "artifacts" / f"full-{mode}-{run_id}"
    bkt_out, mt5_out = out / "bkt", out / "mt5"
    bkt_test_out, query_out, experiment_out = out / "bkt-test", out / "mt5-query-test", out / "ranking"
    if dry_run:
        print("DRY RUN: validation passed. Planned stages: install, tests, BKT, mT5, evaluation, prediction, charts.")
        if dataset["policy_ready"]:
            print("Policy data detected: Dueling DQN can also train.")
        return {"status": "dry_run_passed", "data_status": mode, "counts": dataset["counts"]}

    out.mkdir(parents=True, exist_ok=False)
    if install:
        heading("03 Install")
        command(
            [sys.executable, "-m", "pip", "install", "-q", "-e", ".[model]", "pandas", "matplotlib"],
            cwd=root,
        )
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Colab: Runtime > Change runtime type > GPU")

    heading("04 Validate and run tests")
    if mode == "synthetic":
        hlo(root, "data", "validate", "--directory", dataset["source"])
    command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=root)
    write_statistics(root, mode, out)
    bkt_conf = config_for_run(
        root, "bkt.development.toml" if mode == "synthetic" else "bkt.toml", bkt_out, run_id
    )
    mt5_conf = config_for_run(
        root, "mt5.development.toml" if mode == "synthetic" else "mt5.toml", mt5_out, run_id
    )
    # The repository defaults to a batch of 2. One input at a time is safer on
    # Colab GPUs and still performs genuine parameter updates.
    mt5_text = mt5_conf.read_text(encoding="utf-8")
    mt5_text, changed = re.subn(
        r"(?m)^batch_size\s*=\s*\d+$",
        f"batch_size = {mt5_batch_size}",
        mt5_text,
        count=1,
    )
    if changed != 1:
        raise ValueError("mT5 training config is missing batch_size")
    mt5_conf.write_text(mt5_text, encoding="utf-8")
    synthetic_flag = ["--allow-synthetic"] if mode == "synthetic" else []

    heading("05 Train BKT")
    bkt_train = hlo(root, "train", "bkt", "--config", bkt_conf, *synthetic_flag)
    bkt_model = bkt_out / "model.json"

    heading("06 mT5 preflight")
    hlo(root, "train", "mt5", "--config", mt5_conf, *synthetic_flag, "--check-only")
    heading("07 Download pretrained mT5")
    base_checkpoint = root / "artifacts" / "base" / "mt5-small"
    if not (base_checkpoint / "base_manifest.json").is_file():
        if base_checkpoint.exists() and any(base_checkpoint.iterdir()):
            print("Incomplete earlier mT5 download detected; preserving its files and using a new path.")
            base_checkpoint = root / "artifacts" / "base" / f"mt5-small-{run_id}"
            replacement = base_checkpoint.relative_to(root).as_posix()
            config_text = mt5_conf.read_text(encoding="utf-8").replace(
                'base_checkpoint = "artifacts/base/mt5-small"',
                f'base_checkpoint = "{replacement}"',
            )
            mt5_conf.write_text(config_text, encoding="utf-8")
        hlo(root, "download", "--model", "google/mt5-small", "--output", base_checkpoint)

    heading("08 Fine-tune mT5")
    mt5_train = hlo(root, "train", "mt5", "--config", mt5_conf, *synthetic_flag)
    checkpoint = Path(mt5_train["checkpoint"])
    if not (checkpoint / "manifest.json").is_file():
        raise RuntimeError("Training did not produce a valid mT5 checkpoint manifest")
    write_mt5_history_charts(out, mt5_out)

    policy_result = None
    if dataset["policy_ready"]:
        heading("09 Train optional Dueling Double DQN")
        # The project's policy config always uses data/reviewed and produces artifacts.
        policy_conf = config_for_run(root, "policy.toml", out / "policy", run_id)
        policy_result = hlo(root, "train", "policy", "--config", policy_conf)
    else:
        print("Dueling DQN skipped: no reviewed transition data was provided.")

    heading("10 Evaluate on holdout")
    bkt_test = hlo(
        root, "evaluate", "bkt", "--model", bkt_model,
        "--data", root / "data" / ("development" if mode == "synthetic" else "reviewed") / "assessments.test.jsonl",
        "--output", bkt_test_out,
    )
    query_test = hlo(
        root, "evaluate", "queries", "--generator", "mt5", "--checkpoint", checkpoint,
        "--data", root / "data" / ("development" if mode == "synthetic" else "reviewed") / "questions.test.jsonl",
        "--output", query_out,
    )
    test_questions = dataset["questions"]
    experiment_conf = config_for_run(root, "experiment.toml", experiment_out, run_id)
    if mode == "reviewed":
        # Experiment configuration in the repo defaults to synthetic test questions.
        new_text = experiment_conf.read_text(encoding="utf-8").replace(
            'test = "data/development/questions.test.jsonl"',
            'test = "data/reviewed/questions.test.jsonl"',
        )
        experiment_conf.write_text(new_text, encoding="utf-8")
    experiment = hlo(
        root, "experiment", "--config", experiment_conf, "--generator", "both",
        "--checkpoint", checkpoint, "--output", experiment_out,
    )
    write_evaluation_charts(out, bkt_test_out, experiment_out)

    heading("11 Predictions")
    first = test_questions[0]
    question = first["question"]
    level = first["context"]["level"]
    minutes = str(first["context"]["minutes"])
    prediction = hlo(
        root, "recommend", "--generator", "mt5", "--checkpoint", checkpoint,
        "--question", question, "--level", level, "--minutes", minutes,
        "--learner", f"full-{run_id}", "--bkt-model", bkt_model,
        "--database", out / "learner.sqlite", "--output", out / "prediction_before.json",
    )
    prediction_after = None
    if mode == "synthetic" and prediction.get("target_skills"):
        heading("12 Adaptive prediction demo (synthetic assessments)")
        target_skill = prediction["target_skills"][0]
        now = datetime.now(timezone.utc)
        events = [
            {
                "event_id": f"{run_id}-sample-{i}",
                "learner_id": f"full-{run_id}",
                "skill_id": target_skill,
                "correct": correct,
                "timestamp": (now + timedelta(minutes=i)).isoformat(),
                "data_status": "synthetic",
            }
            for i, correct in enumerate((False, False, True, True, True))
        ]
        events_file = out / "synthetic_prediction_assessments.jsonl"
        events_file.write_text(
            "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
        )
        hlo(
            root, "observe", "--events", events_file, "--database", out / "learner.sqlite",
            "--bkt-model", bkt_model,
        )
        prediction_after = hlo(
            root, "recommend", "--generator", "mt5", "--checkpoint", checkpoint,
            "--question", question, "--level", level, "--minutes", minutes,
            "--learner", f"full-{run_id}", "--bkt-model", bkt_model,
            "--database", out / "learner.sqlite", "--output", out / "prediction_after.json",
        )

    heading("13 Metrics and provenance")
    summary = {
        "status": "completed",
        "run_id": run_id,
        "data_status": mode,
        "note": "Synthetic metrics are not real-world generalization estimates" if mode == "synthetic"
                else "Reviewed data provided; deployment quality requires further validation",
        "counts": dataset["counts"],
        "bkt": {k: bkt_test.get(k) for k in ("n", "accuracy", "auc", "brier", "log_loss")},
        "bkt_validation": bkt_train.get("validation"),
        "mt5": {k: query_test.get(k) for k in ("n", "answer_f1", "valid_rate")},
        "mt5_validation_f1": mt5_train.get("best_validation_nonempty_f1"),
        "mt5_batch_size": mt5_batch_size,
        "ranking": {k: v["metrics"] for k, v in experiment["comparisons"].items()},
        "policy_trained": policy_result is not None,
        "prediction_status": prediction["status"],
        "adaptive_demo_status": prediction_after["status"] if prediction_after else None,
        "bkt_model": str(bkt_model),
        "mt5_checkpoint": str(checkpoint),
        "output_directory": str(out),
    }
    if save_to_drive:
        if not Path("/content").is_dir():
            raise RuntimeError("Google Drive mounting is supported from a Colab session")
        from google.colab import drive
        drive.mount("/content/drive")
        destination = Path("/content/drive/MyDrive/HLO_Full_Training") / run_id
        summary["drive_directory"] = str(destination)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_to_drive:
        shutil.copytree(out, destination, dirs_exist_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command HLO training and evaluation")
    parser.add_argument("--mode", choices=("synthetic", "reviewed"), default="synthetic")
    parser.add_argument("--root", default=None, help="Clone/check out location for recommendation-system")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--require-policy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-to-drive", action="store_true")
    parser.add_argument("--mt5-batch-size", type=int, default=1)
    args = parser.parse_args()
    train_pipeline(
        mode=args.mode, root=args.root, install=not args.no_install,
        require_gpu=not args.allow_cpu, require_policy=args.require_policy,
        dry_run=args.dry_run,
        save_to_drive=args.save_to_drive,
        mt5_batch_size=args.mt5_batch_size,
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
