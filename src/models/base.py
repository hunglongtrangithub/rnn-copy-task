from abc import ABC, abstractmethod
import math
import torch


class Model(torch.nn.Module, ABC):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.linear = torch.nn.Linear(hidden_size, num_classes, bias=True, **factory_kwargs)

    def initialize_parameters(self):
        k = 1 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            torch.nn.init.uniform_(param, -k, k)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size), seq_len, input_size
        Returns:
            outputs: (batch_size), seq_len, num_classes
        """
        pass
