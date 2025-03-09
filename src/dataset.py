import torch


class CopyTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        num_blanks: int,
        vocab_size: int,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_blanks = num_blanks
        self.vocab_size = vocab_size  # vocab tokens are from 0 to vocab_size - 1
        self.blank_token = vocab_size
        self.delimiter_token = vocab_size + 1

        generator = torch.Generator()
        generator.manual_seed(seed)

        self.data = []
        for _ in range(num_samples):
            sequence = torch.randint(0, self.vocab_size, (self.seq_len,), generator=generator)
            input_seq = torch.cat(
                [
                    sequence,
                    torch.full((self.num_blanks,), self.blank_token),
                    torch.full((1,), self.delimiter_token),
                    torch.full((self.seq_len,), self.blank_token),
                ]
            )
            target_seq = torch.cat(
                [
                    torch.full((self.num_blanks + self.seq_len + 1,), self.blank_token),
                    sequence,
                ]
            )
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    params = {
        "num_samples": 10,
        "seq_len": 5,
        "num_blanks": 3,
        "vocab_size": 10,
    }
    d1 = CopyTaskDataset(**params, seed=42)
    d2 = CopyTaskDataset(**params, seed=42)
    dl1 = [d1[i] for i in range(len(d1))]
    dl2 = [d2[i] for i in range(len(d2))]
    # check if the two datasets are the same and the data is consistent with the same seed
    assert all(
        all(
            [
                torch.equal(input1, input2),
                torch.equal(target1, target2),
                input1.shape == target1.shape,
                input2.shape == target2.shape,
            ]
        )
        for (input1, target1), (input2, target2) in zip(dl1, dl2)
    )

    # random split of the dataset
    from torch.utils.data.dataset import random_split

    dataset = CopyTaskDataset(**params, seed=42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    generator2 = torch.Generator().manual_seed(42)
    train1, test1 = random_split(dataset, [train_size, test_size], generator=generator1)
    train2, test2 = random_split(dataset, [train_size, test_size], generator=generator2)
    # check if the two datasets are the same and the data is consistent with the same seed
    assert all(
        all(
            [
                torch.equal(input1, input2),
                torch.equal(target1, target2),
                input1.shape == target1.shape,
                input2.shape == target2.shape,
            ]
        )
        for (input1, target1), (input2, target2) in zip(train1.dataset, train2.dataset)  # type: ignore
    )
