from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

from src.dataset import CopyTaskDataset
from src.models import Model


# One-hot encoding for input tokens
class OneHotEncoder:
    def __init__(self, num_classes):
        self.embedding_matrix = torch.eye(num_classes)

    def encode(self, x):
        return F.embedding(x, self.embedding_matrix.to(x.device))


@dataclass
class ModelConfig:
    vocab_size: int = 10
    input_size: int = 12
    hidden_size: int = 128


@dataclass
class TrainingConfig:
    model_type: str
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    # patience: int = 5  # Early stopping patience
    clip_grad_norm: float = 5.0
    num_train_samples: int = 800
    num_val_samples: int = 100
    num_test_samples: int = 100
    num_blanks: int = 10
    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class Trainer:
    def __init__(
        self,
        model: Model,
        config: TrainingConfig,
        model_config: ModelConfig,
        seq_len: int,
    ):
        self.model = model
        self.config = config
        self.model_config = model_config
        self.seq_len = seq_len

        # Create datasets
        self.train_dataset = CopyTaskDataset(
            config.num_train_samples,
            seq_len,
            config.num_blanks,
            model_config.vocab_size,
            seed=config.seed,
        )
        self.val_dataset = CopyTaskDataset(
            config.num_val_samples,
            seq_len,
            config.num_blanks,
            model_config.vocab_size,
            seed=config.seed + 1,
        )
        self.test_dataset = CopyTaskDataset(
            config.num_test_samples,
            seq_len,
            config.num_blanks,
            model_config.vocab_size,
            seed=config.seed + 2,
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Number of classes = vocab_size + 2 (for blank token and delimiter token)
        self.num_classes = model_config.vocab_size + 2
        self.encoder = OneHotEncoder(self.num_classes)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.train_dataset.blank_token
        )  # Ignore blank tokens in loss

        self.model = self.model.to(config.device)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float("inf")
        # self.patience_counter = 0

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            logger.debug(f"inputs.shape: {inputs.shape}")
            logger.debug(f"targets.shape: {targets.shape}")

            inputs_one_hot = self.encoder.encode(inputs)
            logger.debug(f"inputs_one_hot.shape; {inputs_one_hot.shape}")

            self.optimizer.zero_grad()
            # Forward pass to get logits
            outputs = self.model(inputs_one_hot)
            logger.debug(f"outputs.shape: {outputs.shape}")

            # Reshape to feed into CrossEntropyLoss
            outputs = outputs.reshape(-1, self.num_classes)
            targets = targets.reshape(-1)
            logger.debug(f"outputs.shape (reshaped): {outputs.shape}")
            logger.debug(f"targets.shape (reshaped): {targets.shape}")

            # Take the loss across the entire batch
            loss = self.criterion(outputs, targets)

            loss.backward()
            # TODO: Is this mandatory?
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Calculate accuracy. Softmax then argmax
            pred = F.softmax(outputs, dim=1).argmax(dim=1)
            logger.debug(f"pred.shape: {pred.shape} - targets.shape: {targets.shape}")
            correct = (pred == targets).sum().item()
            total_correct += correct
            total_tokens += targets.numel()

            if batch_idx % 5 == 0:
                print(
                    f"Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f} - Accuracy: {total_correct / max(1, total_tokens):.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / max(1, total_tokens)

        return avg_loss, accuracy

    def validate(self) -> tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                inputs_one_hot = self.encoder.encode(inputs)

                outputs = self.model(inputs_one_hot)

                outputs = outputs.reshape(-1, self.num_classes)
                targets = targets.reshape(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                pred = F.softmax(outputs, dim=1).argmax(dim=1)
                correct = (pred == targets).sum().item()
                total_correct += correct
                total_tokens += targets.numel()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / max(1, total_tokens)

        return avg_loss, accuracy

    def test(self) -> float:
        """Test the model"""
        self.model.eval()
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                inputs_one_hot = self.encoder.encode(inputs)

                outputs = self.model(inputs_one_hot)

                outputs = outputs.reshape(-1, self.num_classes)
                targets = targets.reshape(-1)

                pred = F.softmax(outputs, dim=1).argmax(dim=1)
                correct = (pred == targets).sum().item()
                total_correct += correct
                total_tokens += targets.numel()

        accuracy = total_correct / max(1, total_tokens)

        return accuracy

    def train(self) -> dict:
        """Train the model with early stopping"""
        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # Check for early stopping
            if val_loss < self.best_val_loss:
                print("Saving best model")
                self.best_val_loss = val_loss
                # self.patience_counter = 0
                # Save best model
                self.best_model_state = {
                    key: value.cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            # else:
            #     self.patience_counter += 1
            #     if self.patience_counter >= self.config.patience:
            #         print(f"Early stopping after {epoch + 1} epochs")
            #         break

        print("Training complete")
        print("Loading best model for testing")
        # Load best model for testing
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        # Test the model
        test_acc = self.test()
        print(f"Test Accuracy: {test_acc:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "test_acc": test_acc,
            "final_train_acc": self.train_accs[-1],
            "final_val_acc": self.val_accs[-1],
        }
