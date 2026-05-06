
import argparse
import shutil
from pathlib import Path

from src.train import run_experiment

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
MODEL_TYPES = ["baseline", "dual"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Drive path to copy checkpoints and metrics into")
    parser.add_argument("--csv_path", default="dataset_index.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    for model_type in MODEL_TYPES:
        for held_out in METHODS:
            train_methods = [m for m in METHODS if m != held_out]
            experiment_name = f"{model_type}_holdout_{held_out}"

            print("=" * 80)
            print(f"Model: {model_type}  |  Held-out: {held_out}")
            print(f"Training on: {train_methods}")
            print("=" * 80)

            run_experiment(
                experiment_name=experiment_name,
                csv_path=args.csv_path,
                train_methods=train_methods,
                val_methods=train_methods,
                test_methods=[held_out],
                batch_size=32,
                epochs=10,
                max_train_samples=10000,
                max_val_samples=1500,
                max_test_samples=1500,
                model_type=model_type,
                drive_checkpoint_dir="/content/drive/MyDrive/deepfake-detection/vriddhi_results/checkpoints",
            )

            for src, dst_dir in [
                (Path("results/checkpoints") / f"{experiment_name}.pt", output_dir / "checkpoints"),
                (Path("results/metrics") / f"{experiment_name}.json", output_dir / "metrics"),
            ]:
                if src.exists():
                    shutil.copy2(src, dst_dir / src.name)
                    print(f"Copied {src.name} → {dst_dir}")

    write_summary_csv(output_dir)


def write_summary_csv(output_dir: Path):
    import csv
    import json

    metrics_dir = output_dir / "metrics"
    rows = []

    for model_type in MODEL_TYPES:
        for held_out in METHODS:
            experiment_name = f"{model_type}_holdout_{held_out}"
            json_path = metrics_dir / f"{experiment_name}.json"

            if not json_path.exists():
                print(f"Missing: {json_path}, skipping")
                continue

            with open(json_path) as f:
                data = json.load(f)

            tm = data["test_metrics"]
            rows.append({
                "experiment_name": experiment_name,
                "model_type": model_type,
                "held_out": held_out,
                "accuracy": tm["accuracy"],
                "macro_f1": tm["macro_f1"],
                "auc": tm["auc"],
            })

    summary_path = metrics_dir / "vriddhi_summary.csv"
    fieldnames = ["experiment_name", "model_type", "held_out", "accuracy", "macro_f1", "auc"]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary ({len(rows)} rows) → {summary_path}")


if __name__ == "__main__":
    main()
