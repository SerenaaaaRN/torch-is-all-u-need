import numpy as np


class MSELoss:
    def __init__(self):
        self.predicted = None
        self.target = None

    def __call__(self, predicted, target):
        predicted = np.asarray(predicted, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if predicted.shape != target.shape:
            raise ValueError(
                f"Shape prediksi {predicted.shape} tidak sama dengan target {target.shape}"
            )
        self.predicted = predicted
        self.target = target
        self.loss = float(np.mean((predicted - target) ** 2))
        return self.loss

    def backward(self):
        return 2.0 * (self.predicted - self.target) / self.predicted.size


class BCELoss:
    def __init__(self, eps=1e-7):
        self.eps = eps
        self.predicted = None
        self.target = None

    def __call__(self, predicted, target):
        predicted = np.asarray(predicted, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if predicted.shape != target.shape:
            raise ValueError(
                f"Shape prediksi {predicted.shape} tidak sama dengan target {target.shape}"
            )
        self.predicted = np.clip(predicted, self.eps, 1.0 - self.eps)
        self.target = target
        self.loss = float(
            -np.mean(
                target * np.log(self.predicted)
                + (1.0 - target) * np.log(1.0 - self.predicted)
            )
        )
        return self.loss

    def backward(self):
        return (
            -self.target / self.predicted
            + (1.0 - self.target) / (1.0 - self.predicted)
        ) / self.predicted.size


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.target = None

    def __call__(self, logits, target):
        logits = np.asarray(logits, dtype=np.float64)
        target = np.asarray(target).reshape(-1).astype(np.int64)
        if logits.ndim != 2:
            raise ValueError(f"Logits harus berbentuk (batch, kelas), dapat {logits.shape}")
        if target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Jumlah target {target.shape[0]} tidak sama dengan batch {logits.shape[0]}"
            )

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        self.probs = probs
        self.target = target
        self.loss = float(
            -np.mean(np.log(probs[np.arange(logits.shape[0]), target] + 1e-12))
        )
        return self.loss

    def backward(self):
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.target] -= 1.0
        grad /= batch_size
        return grad
