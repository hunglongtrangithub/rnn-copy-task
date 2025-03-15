from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_dir: Path):
    """Load results from JSON file"""
    with open(results_dir / "experiment_results.json", "r") as f:
        results = json.load(f)
    return results


def analyze_long_term_memory(results, sequence_lengths, model_types):
    """Analyze and print the long-term memory capability of each model."""
    print("\nAnalyzing Long-Term Memory Capability:")
    for seq_len in sequence_lengths:
        print(f"\nSequence Length: {seq_len}")
        for model_type in model_types:
            mean_test_acc = results[str(seq_len)][model_type]["mean_test_acc"]
            std_test_acc = results[str(seq_len)][model_type]["std_test_acc"]
            print(
                f"{model_type}: Mean Test Accuracy = {mean_test_acc:.4f} ± {std_test_acc:.4f}"
            )


def analyze_multiplicative_effects(results, sequence_lengths):
    """Analyze and print observations on multiplicative effects."""
    print("\nAnalyzing Multiplicative Effects:")
    for seq_len in sequence_lengths:
        print(f"\nSequence Length: {seq_len}")
        lstm_acc = results[str(seq_len)]["LSTM"]["mean_test_acc"]
        mul_lstm_acc = results[str(seq_len)]["MultiplicativeLSTM"]["mean_test_acc"]
        gru_acc = results[str(seq_len)]["GRU"]["mean_test_acc"]
        mul_gru_acc = results[str(seq_len)]["MultiplicativeGRU"]["mean_test_acc"]

        print(f"LSTM vs Multiplicative LSTM: {lstm_acc:.4f} vs {mul_lstm_acc:.4f}")
        print(f"GRU vs Multiplicative GRU: {gru_acc:.4f} vs {mul_gru_acc:.4f}")

        # Compare convergence speed (epochs to converge)
        lstm_epochs = results[str(seq_len)]["LSTM"]["mean_epochs"]
        mul_lstm_epochs = results[str(seq_len)]["MultiplicativeLSTM"]["mean_epochs"]
        gru_epochs = results[str(seq_len)]["GRU"]["mean_epochs"]
        mul_gru_epochs = results[str(seq_len)]["MultiplicativeGRU"]["mean_epochs"]

        print(
            f"LSTM vs Multiplicative LSTM Epochs: {lstm_epochs:.2f} vs {mul_lstm_epochs:.2f}"
        )
        print(
            f"GRU vs Multiplicative GRU Epochs: {gru_epochs:.2f} vs {mul_gru_epochs:.2f}"
        )

        # Compare training stability (standard deviation of test accuracy)
        lstm_std = results[str(seq_len)]["LSTM"]["std_test_acc"]
        mul_lstm_std = results[str(seq_len)]["MultiplicativeLSTM"]["std_test_acc"]
        gru_std = results[str(seq_len)]["GRU"]["std_test_acc"]
        mul_gru_std = results[str(seq_len)]["MultiplicativeGRU"]["std_test_acc"]

        print(
            f"LSTM vs Multiplicative LSTM Stability: {lstm_std:.4f} vs {mul_lstm_std:.4f}"
        )
        print(
            f"GRU vs Multiplicative GRU Stability: {gru_std:.4f} vs {mul_gru_std:.4f}"
        )


def analyze_longest_sequence(results, sequence_lengths):
    """Analyze and print which model handles the longest sequence best."""
    longest_seq = max(sequence_lengths)
    print(f"\nAnalyzing Performance at Longest Sequence Length ({longest_seq}):")

    best_model = None
    best_acc = 0.0

    for model_type in results[str(longest_seq)]:
        mean_test_acc = results[str(longest_seq)][model_type]["mean_test_acc"]
        print(f"{model_type}: Mean Test Accuracy = {mean_test_acc:.4f}")
        if mean_test_acc > best_acc:
            best_acc = mean_test_acc
            best_model = model_type

    print(
        f"\nBest Model for Longest Sequence: {best_model} with Accuracy = {best_acc:.4f}"
    )


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
            accuracies.append(results[str(seq_len)][model_type]["mean_test_acc"])
            std_errors.append(results[str(seq_len)][model_type]["std_test_acc"])
            train_times.append(results[str(seq_len)][model_type]["mean_train_time"])
            epochs_to_converge.append(results[str(seq_len)][model_type]["mean_epochs"])

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


def main():
    """Main function to run analysis on saved results"""
    model_types = ["LSTM", "MultiplicativeLSTM", "GRU", "MultiplicativeGRU"]
    sequence_lengths = [100, 200, 500, 1000]

    reports_dir = Path(__file__).parents[1] / "reports"

    # Load results
    results = load_results(reports_dir)

    # Analyze results
    analyze_long_term_memory(results, sequence_lengths, model_types)
    analyze_multiplicative_effects(results, sequence_lengths)
    analyze_longest_sequence(results, sequence_lengths)

    # Plot comparative results
    plot_comparative_results(results, sequence_lengths, model_types, reports_dir)

    print("\nAnalysis completed. Plots saved to disk.")


if __name__ == "__main__":
    main()
