import numpy as np

from tinytorch import (
    Adam,
    BatchNorm,
    CrossEntropyLoss,
    DataLoader,
    Dropout,
    Linear,
    ReLU,
    Sequential,
)


def make_spiral_data(points_per_class=100, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    inputs = []
    targets = []

    for class_index in range(3):
        radius = np.linspace(0.0, 1.0, points_per_class)
        theta = np.linspace(
            class_index * 4.0, (class_index + 1) * 4.0, points_per_class
        ) + rng.normal(0.0, noise, points_per_class)
        inputs.append(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]))
        targets.append(np.full(points_per_class, class_index, dtype=np.int64))

    return np.vstack(inputs), np.concatenate(targets)


def build_model():
    return Sequential(
        Linear(2, 64),
        BatchNorm(64),
        ReLU(),
        Dropout(0.1),
        Linear(64, 64),
        BatchNorm(64),
        ReLU(),
        Dropout(0.1),
        Linear(64, 3),
    )


def train():
    np.random.seed(42)
    inputs, targets = make_spiral_data()

    indices = np.random.permutation(len(inputs))
    split = int(len(inputs) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_data = list(zip(inputs[train_indices], targets[train_indices]))
    test_inputs = inputs[test_indices]
    test_targets = targets[test_indices]

    model = build_model()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    loader = DataLoader(train_data, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(300):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)

            optimizer.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()

            total_loss += loss * len(batch_inputs)
            total_correct += int((logits.argmax(axis=1) == batch_targets).sum())
            total_samples += len(batch_inputs)

        if epoch % 50 == 0 or epoch == 299:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples * 100
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%"
            )

    model.eval()
    predictions = model(test_inputs).argmax(axis=1)
    accuracy = float((predictions == test_targets).mean()) * 100
    correct = int((predictions == test_targets).sum())
    print(f"\nTest Accuracy: {accuracy:.1f}% ({correct}/{len(test_targets)})")
    return model, accuracy


if __name__ == "__main__":
    train()
