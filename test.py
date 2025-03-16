import torch
import random


def test_broadcast_mul():
    x = torch.arange(3 * 4).view(3, 4)
    y = torch.tensor([0, 1, 0, 1])
    z = x * y
    print(z)


def test_random(seed=42):
    random.seed(seed)
    return [random.randint(1, 10) for _ in range(5)]


class InMemoryDataset:
    def __init__(self, seed=42):
        random.seed(seed)
        self.data = [random.randint(1, 10) for _ in range(5)]
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return 5


class TemporaryDataset:
    def __init__(self, seed=42):
        self.random = random.Random(seed)
        self.len = 5

    def __getitem__(self, _):
        return self.random.randint(1, 10)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    l1 = test_random(2)
    l2 = test_random(2)
    print(l1, l2)
    print(l1 == l2)
    d1 = TemporaryDataset(2)
    d2 = TemporaryDataset(2)
    dl1 = [d1[i] for i in range(len(d1))]
    dl2 = [d2[i] for i in range(len(d2))]
    print(dl1, dl2)
    print(dl1 == dl2)
