import torch


def test_broadcast_mul():
    x = torch.arange(3 * 4).view(3, 4)
    y = torch.tensor([0, 1, 0, 1])
    z = x * y
    print(z)


if __name__ == "__main__":
    test_broadcast_mul()
