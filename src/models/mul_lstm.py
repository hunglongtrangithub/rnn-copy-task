import math
import torch


class MulLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_i = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.W_f = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.W_o = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.W_c = torch.nn.Parameter(torch.empty((input_size, hidden_size), **factory_kwargs))
        self.W_m = torch.nn.Parameter(torch.empty((input_size, input_size), **factory_kwargs))

        self.U_i = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        self.U_f = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        self.U_o = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        self.U_c = torch.nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        self.U_m = torch.nn.Parameter(torch.empty((hidden_size, input_size), **factory_kwargs))

        self.b_i = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.b_f = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.b_o = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.b_c = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.b_m = torch.nn.Parameter(torch.empty(input_size, **factory_kwargs))

        self.linear = torch.nn.Linear(hidden_size, num_classes, bias=True, **factory_kwargs)

        k = 1 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            torch.nn.init.uniform_(param, -k, k)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size), seq_len, input_size
        Returns:
            outputs: (batch_size), seq_len, num_classes
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected input to be 2D or 3D, got {x.dim()}D instead")
        is_batched = x.dim() == 3
        x = torch.unsqueeze(x, 0) if x.dim() == 2 else x

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        h_t = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)
        c_t = torch.zeros((batch_size, self.hidden_size), device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            m_t = x_t @ self.W_m + h_t @ self.U_m + self.b_m  # (input_size)
            x_t = x_t * m_t  # (batch_size, input_size)

            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)  # (batch_size, hidden_size)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)  # (batch_size, hidden_size)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)  # (batch_size, hidden_size)
            c_tilde_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)  # (batch_size, hidden_size)

            c_t = f_t * c_t + i_t * c_tilde_t  # (batch_size, hidden_size)
            h_t = o_t * torch.tanh(c_t)  # (batch_size, hidden_size)

            outputs.append(torch.unsqueeze(h_t, 1))  # (batch_size, 1, hidden_size)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        outputs = self.linear(outputs)  # (batch_size, seq_len, num_labels)

        if not is_batched:
            outputs = torch.squeeze(outputs, 0)  # (seq_len, num_classes)
        return outputs


if __name__ == "__main__":
    # Test the LSTM implementation
    input_size = 10
    hidden_size = 20
    num_classes = 5
    batch_size = 5
    seq_len = 15

    model = MulLSTM(input_size, hidden_size, num_classes)
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    print(output.shape)  # Expected shape: (batch_size, seq_len, hidden_size)
