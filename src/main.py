from pathlib import Path
import time
import json
import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from src.models import LSTM, MultiplicativeLSTM, GRU, MultiplicativeGRU
from src.train import Trainer, ModelConfig, TrainingConfig, plot_metrics

# Ensure reproducibility
SEED = 42


# Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(model_types: list[str], sequence_lengths: list[int], n_trials: int, save_dir: Path):
    """Run experiments for all model types and sequence lengths"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_config = ModelConfig()

    results = {}

    for seq_len in sequence_lengths:
        results[seq_len] = {}
        for model_type in model_types:
            print(f"\n{'=' * 50}")
            print(f"Running {model_type} with sequence length {seq_len}")
            print(f"{'=' * 50}")

            # Store results for multiple trials
            trial_results = []

            for trial in range(n_trials):
                print(f"\nTrial {trial + 1}/{n_trials}")

                # Set different seed for each trial
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
                    model_type=model_type,
                    seed=trial_seed,
                    device=device,
                )

                # Start training timer
                start_time = time.time()

                # Train model
                trainer = Trainer(model, training_config, model_config, seq_len)
                metrics = trainer.train()

                # End training timer
                train_time = time.time() - start_time

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

                # Plot metrics for this trial
                plots_dir = save_dir / str(seq_len) / model_type
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_metrics(metrics, f"{model_type}_trial{trial + 1}", seq_len, plots_dir)

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


def plot_comparative_results(results, sequence_lengths, model_types, save_dir):
    """Plot comparative results across different models and sequence lengths"""
    # Prepare data for plots
    seq_lens = []
    model_names = []
    accuracies = []
    std_errors = []
    train_times = []
    epochs_to_converge = []

    for seq_len in sequence_lengths:
        for model_type in model_types:
            seq_lens.append(seq_len)
            model_names.append(model_type)
            accuracies.append(results[seq_len][model_type]["mean_test_acc"])
            std_errors.append(results[seq_len][model_type]["std_test_acc"])
            train_times.append(results[seq_len][model_type]["mean_train_time"])
            epochs_to_converge.append(results[seq_len][model_type]["mean_epochs"])

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "Sequence Length": seq_lens,
            "Model": model_names,
            "Accuracy": accuracies,
            "Std Error": std_errors,
            "Training Time": train_times,
            "Epochs": epochs_to_converge,
        }
    )

    # Plot 1: Accuracy vs Sequence Length
    plt.figure(figsize=(12, 8))

    for model_type in model_types:
        model_data = df[df["Model"] == model_type]
        plt.errorbar(
            model_data["Sequence Length"],
            model_data["Accuracy"],
            yerr=model_data["Std Error"],
            marker="o",
            linestyle="-",
            label=model_type,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Test Accuracy")
    plt.title("Model Accuracy vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "accuracy_vs_seq_length.png")
    plt.close()

    # Plot 2: Training Time vs Sequence Length
    plt.figure(figsize=(12, 8))

    for model_type in model_types:
        model_data = df[df["Model"] == model_type]
        plt.plot(
            model_data["Sequence Length"],
            model_data["Training Time"],
            marker="o",
            linestyle="-",
            label=model_type,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "train_time_vs_seq_length.png")
    plt.close()

    # Plot 3: Epochs to Converge vs Sequence Length
    plt.figure(figsize=(12, 8))

    for model_type in model_types:
        model_data = df[df["Model"] == model_type]
        plt.plot(
            model_data["Sequence Length"],
            model_data["Epochs"],
            marker="o",
            linestyle="-",
            label=model_type,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Epochs to Converge")
    plt.title("Convergence Speed vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "epochs_vs_seq_length.png")
    plt.close()

    # Create a bar chart comparing accuracy at the longest sequence length
    longest_seq = max(sequence_lengths)
    longest_seq_data = df[df["Sequence Length"] == longest_seq]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        longest_seq_data["Model"],
        longest_seq_data["Accuracy"],
        yerr=longest_seq_data["Std Error"],
        alpha=0.7,
        capsize=5,
    )

    plt.xlabel("Model Type")
    plt.ylabel("Test Accuracy")
    plt.title(f"Model Accuracy at Sequence Length {longest_seq}")
    plt.ylim(0, 1.0)

    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.savefig(save_dir / f"accuracy_comparison_seq{longest_seq}.png")
    plt.close()


def save_results(results):
    """Save results to JSON file"""
    results_dir = Path(__file__).parents[1] / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)

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
    with open(results_dir / "experiment_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)


def main():
    """Main function to run all experiments"""
    model_types = ["LSTM", "MultiplicativeLSTM", "GRU", "MultiplicativeGRU"]
    sequence_lengths = [100, 200, 500, 1000]
    n_trials = 3

    reports_dir = Path(__file__).parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = run_experiment(model_types, sequence_lengths, n_trials, reports_dir / "plots")

    # Plot comparative results
    plot_comparative_results(results, sequence_lengths, model_types, reports_dir)

    # Save results
    save_results(results)

    print("\nExperiments completed. Results and plots saved.")


if __name__ == "__main__":
    main()
