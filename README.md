# LSTM and GRU vs. Multiplicative Variants on the Copy Task

This project explores four different RNN architectures (LSTM, Multiplicative LSTM, GRU, Multiplicative GRU) on the Copy Task, a benchmark that tests a model’s ability to memorize and reproduce sequences after a delay. Our experiments assess model performance at sequence lengths of 100, 200, 500, and 1000 tokens, and compare training time, convergence speed, and final accuracy.

All training, testing, and analysis scripts are located in the `src/` directory. Results, logs, and figures are stored in `reports/`. A detailed report describing implementation details and findings is available in `report.md`.

---

## Key Findings

- **Near-Random Accuracy**  
  All four models hovered around 10.0% accuracy (random guessing for a 10-token vocabulary), indicating they largely failed to learn the copy task under these experimental settings.

- **Long-Term Memory Challenge**  
  Extending the sequence length from 100 up to 1000 did not meaningfully change performance; none of the models surpassed ~0.10 accuracy.

- **Comparing Multiplicative vs. Standard Gating**  
  Neither multiplicative LSTM nor multiplicative GRU demonstrated a consistent performance advantage. Convergence patterns were similar across all variants.

- **Training Time**  
  As expected, GRU was the fastest to train, followed by LSTM, multiplicative GRU, and multiplicative LSTM. At sequence length 1000, LSTM and mLSTM required nearly twice the training time of GRU/mGRU.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- [UV](https://github.com/astral-sh/uv) for dependency management

### Install Required Libraries

Use `uv` to install dependencies:

```bash
uv sync
```

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

---

## Project Structure

```
.
├── reports/                         # Contains logs, figures, and result JSON for experiments
│   ├── experiment_logs/             # Loss/accuracy plots for individual trials
│   ├── experiment_results.json      # Aggregated accuracy/time/epoch data
│   ├── accuracy_vs_seq_length.png
│   ├── train_time_vs_seq_length.png
│   ├── epochs_vs_seq_length.png
│   ├── accuracy_comparison_seq1000.png
│   └── REPORT.md                    # Detailed report describing results and analyses
├── src/                             # Source code directory
│   ├── dataset.py                   # CopyTaskDataset class for generating the copy task data
│   ├── trainer.py                   # Training loop, data loading, logging, and test routines
│   ├── models/                      # RNN model definitions
│   └── scripts/                     # Scripts to run experiment and postprocessing of results
├── README.md                        # Project overview and instructions
└── .gitignore
```

---

## Running the Code

Below are examples of the main commands. Please customize these commands based on your specific file locations and naming:

1. **Run Experiments**

   ```bash
   python -m src.scripts.experiment
   ```

   This will train each of the four architectures (LSTM, mLSTM, GRU, mGRU) on sequence lengths {100, 200, 500, 1000}, saving the logs and JSON results.

2. **Analyze and Plot**

   ```bash
   python -m src.scripts.analyze
   ```

   This will parse the JSON results, compute statistics, and generate plots (accuracy/time/epochs) in the `reports/` directory.

---

## Results

### Accuracy vs. Sequence Length

- Across **T = 100 → 1000**, test accuracy remained at ~10%, highlighting the difficulty of learning delayed copy with standard RNN training.

### Training Time

- **Order**: GRU < LSTM < mGRU < mLSTM.
- At T=1000, LSTM/mLSTM training time was nearly double that of GRU/mGRU.

### Discussion Highlights

- **Long-Term Memory Failure**: None of the four models learned to copy sequences effectively at the given hyperparameters.
- **No Clear Multiplicative Boost**: mLSTM/mGRU didn’t significantly outperform or converge faster than standard LSTM/GRU.
- **Potential Future Directions**: More advanced gating strategies, alternative hyperparameters, or attention mechanisms might be required to solve the copy task for large T.

_(See the full discussion in `reports/report.md`.)_

---

## Future Work

- **Hyperparameter Tuning**: Investigate different learning rates, optimizers, or random seeds to see if any combination can break past the random 10% accuracy.
- **Curriculum Learning**: Start from very short sequences (T < 50) and gradually increase the length, helping the model incrementally adapt to longer memories.
- **Architectural Variations**: Explore orthogonal/transformer-based solutions or additional gating modifications to improve gradient flow for extended sequence lengths.
- **Debugging**: Confirm correct data generation, ensure no hidden bugs in data preprocessing or training loops.
