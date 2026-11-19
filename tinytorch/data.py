import numpy as np


class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True, drop_last=False):
        if batch_size <= 0:
            raise ValueError("batch_size harus lebih besar dari 0")
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch = [self.data[i] for i in batch_indices]
            inputs = np.stack(
                [np.asarray(item[0], dtype=np.float64) for item in batch]
            )
            targets = np.stack([np.asarray(item[1]) for item in batch])
            yield inputs, targets

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        return (len(self.data) + self.batch_size - 1) // self.batch_size
