
from src.train import run_experiment

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def main():
    for held_out in METHODS:
        train_methods = [m for m in METHODS if m != held_out]

        print("=" * 80)
        print(f"Held-out method: {held_out}")
        print(f"Training methods: {train_methods}")
        print("=" * 80)

        run_experiment(
            experiment_name=f"dual_holdout_{held_out}",
            model_type="dual",
            train_methods=train_methods,
            val_methods=train_methods,
            test_methods=[held_out],
            epochs=3,
            max_train_samples=5000,
            max_val_samples=1500,
            max_test_samples=1500,
            batch_size=32,
        )


if __name__ == "__main__":
    main()
