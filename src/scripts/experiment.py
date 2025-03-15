from pathlib import Path
import time
import json
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from src.models import LSTM, MultiplicativeLSTM, GRU, MultiplicativeGRU
from src.trainer import Trainer, ModelConfig, TrainingConfig


# Ensure reproducibility
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    model_types: list[str], sequence_lengths: list[int], n_trials: int, save_dir: Path
):
    """Run experiments for all model types and sequence lengths"""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    model_config = ModelConfig()
    results = {}

    for seq_len in sequence_lengths:
        results[seq_len] = {}
        for model_type in model_types:
            print(f"\n{'=' * 50}")
            print(f"Running {model_type} with sequence length {seq_len}")
            print(f"{'=' * 50}")

            trial_results = []
            for trial in range(n_trials):
                print(f"\nTrial {trial + 1}/{n_trials}")
                trial_seed = SEED + trial
                set_seed(trial_seed)

                # Create model
                if model_type == "LSTM":
                    model = LSTM(
                        model_config.input_size,
                        model_config.hidden_size,
                        model_config.vocab_size + 2,
                    )
                elif model_type == "MultiplicativeLSTM":
                    model = MultiplicativeLSTM(
                        model_config.input_size,
                        model_config.hidden_size,
                        model_config.vocab_size + 2,
                    )
                elif model_type == "GRU":
                    model = GRU(
                        model_config.input_size,
                        model_config.hidden_size,
                        model_config.vocab_size + 2,
                    )
                elif model_type == "MultiplicativeGRU":
                    model = MultiplicativeGRU(
                        model_config.input_size,
                        model_config.hidden_size,
                        model_config.vocab_size + 2,
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Configure training
                training_config = TrainingConfig(
                    model_type=model_type, seed=trial_seed, device=device
                )

                # Train model
                start_time = time.time()
                # NOTE: test sequence length is set to seq_len + 100 to test generalization
                trainer = Trainer(
                    model, training_config, model_config, seq_len, seq_len + 100
                )
                metrics = trainer.train()
                train_time = time.time() - start_time

                # Plot metrics for this trial
                plots_dir = save_dir / str(seq_len) / model_type
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_metrics(
                    metrics, f"{model_type}_trial{trial + 1}", seq_len, plots_dir
                )

                # Store results
                trial_result = {
                    "test_acc": metrics["test_acc"],
                    "train_time": train_time,
                    "epochs": len(metrics["train_losses"]),
                    "final_train_acc": metrics["train_accs"][-1],
                    "final_val_acc": metrics["val_accs"][-1],
                    "best_val_loss": trainer.best_val_loss,
                    "train_losses": metrics["train_losses"],
                    "val_losses": metrics["val_losses"],
                    "train_accs": metrics["train_accs"],
                    "val_accs": metrics["val_accs"],
                }
                trial_results.append(trial_result)

            # Calculate statistics across trials
            test_accs = [result["test_acc"] for result in trial_results]
            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)

            train_times = [result["train_time"] for result in trial_results]
            mean_train_time = np.mean(train_times)

            epochs = [result["epochs"] for result in trial_results]
            mean_epochs = np.mean(epochs)

            # Store aggregated results
            results[seq_len][model_type] = {
                "mean_test_acc": mean_test_acc,
                "std_test_acc": std_test_acc,
                "mean_train_time": mean_train_time,
                "mean_epochs": mean_epochs,
                "trial_results": trial_results,
            }

            print(f"Results for {model_type} at sequence length {seq_len}:")
            print(f"Mean test accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
            print(f"Mean training time: {mean_train_time:.2f} seconds")
            print(f"Mean epochs until convergence: {mean_epochs:.2f}")

    return results


def plot_metrics(metrics_dict: dict, model_type: str, seq_len: int, save_dir: Path):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(metrics_dict["train_losses"], label="Train Loss")
    plt.plot(metrics_dict["val_losses"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_type} - Loss (Seq Len: {seq_len})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics_dict["train_accs"], label="Train Acc")
    plt.plot(metrics_dict["val_accs"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_type} - Accuracy (Seq Len: {seq_len})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"{model_type}_seq{seq_len}.png")
    plt.close()


def save_results(results, save_dir: Path):
    """Save results to JSON file"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to serializable format
    serializable_results = {}
    for seq_len, seq_results in results.items():
        serializable_results[str(seq_len)] = {}
        for model_type, model_results in seq_results.items():
            serializable_results[str(seq_len)][model_type] = {
                "mean_test_acc": float(model_results["mean_test_acc"]),
                "std_test_acc": float(model_results["std_test_acc"]),
                "mean_train_time": float(model_results["mean_train_time"]),
                "mean_epochs": float(model_results["mean_epochs"]),
                "trial_results": [
                    {
                        "test_acc": float(trial["test_acc"]),
                        "train_time": float(trial["train_time"]),
                        "epochs": int(trial["epochs"]),
                        "final_train_acc": float(trial["final_train_acc"]),
                        "final_val_acc": float(trial["final_val_acc"]),
                        "best_val_loss": float(trial["best_val_loss"]),
                        "train_losses": [float(loss) for loss in trial["train_losses"]],
                        "val_losses": [float(loss) for loss in trial["val_losses"]],
                        "train_accs": [float(acc) for acc in trial["train_accs"]],
                        "val_accs": [float(acc) for acc in trial["val_accs"]],
                    }
                    for trial in model_results["trial_results"]
                ],
            }

    # Save to file
    with open(save_dir / "experiment_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)


def main():
    """Main function to run all experiments and save results"""
    model_types = ["LSTM", "MultiplicativeLSTM", "GRU", "MultiplicativeGRU"]
    sequence_lengths = [100, 200, 500, 1000]
    n_trials = 3

    reports_dir = Path(__file__).parents[2] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = run_experiment(
        model_types, sequence_lengths, n_trials, reports_dir / "experiment_plots"
    )

    # Save results
    save_results(results, reports_dir)

    print("\nExperiments completed. Results saved to disk.")


if __name__ == "__main__":
    main()
