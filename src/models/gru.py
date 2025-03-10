import torch

from src.models.base import Model


class GRU(Model):
    def __init__(self, input_size, hidden_size, num_classes, device=None, dtype=None):
        super().__init__(input_size, hidden_size, num_classes, device=device, dtype=dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.W_z = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.W_r = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.W_h = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))

        self.U_z = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        self.U_r = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        self.U_h = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))

        self.b_z = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.b_r = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.b_h = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.initialize_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected input to be 2D or 3D, got {x.dim()}D instead")
        is_batched = x.dim() == 3
        x = torch.unsqueeze(x, 0) if x.dim() == 2 else x

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        h_t = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            z_t = torch.sigmoid(x_t @ self.W_z + h_t @ self.U_z + self.b_z)  # (batch_size, hidden_size)
            r_t = torch.sigmoid(x_t @ self.W_r + h_t @ self.U_r + self.b_r)  # (batch_size, hidden_size)
            h_tilde_t = torch.tanh(x_t @ self.W_h + (r_t * h_t) @ self.U_h + self.b_h)  # (batch_size, hidden_size)

            h_t = (1 - z_t) * h_t + z_t * h_tilde_t  # (batch_size, hidden_size)

            outputs.append(torch.unsqueeze(h_t, 1))  # (batch_size, 1, hidden_size)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        outputs = self.linear(outputs)  # (batch_size, seq_len, num_classes)

        if not is_batched:
            outputs = torch.squeeze(outputs, 0)  # (seq_len, num_classes)
        return outputs


if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    num_classes = 5
    batch_size = 5
    seq_len = 15

    model = GRU(input_size, hidden_size, num_classes)
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    print(output.shape)  # Expected shape: (batch_size, seq_len, hidden_size)
