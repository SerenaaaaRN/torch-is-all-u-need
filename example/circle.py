import numpy as np

from tinytorch import Adam, BCELoss, DataLoader, Linear, ReLU, Sequential, Sigmoid


def make_circle_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-2, 2, size=(n, 2))
    labels = (points[:, 0] ** 2 + points[:, 1] ** 2 < 1.5).astype(np.float64)
    return points, labels.reshape(-1, 1)


def train():
    np.random.seed(42)
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data[0]) * 0.8)
    train_data = list(zip(data[0][:split], data[1][:split]))
    test_inputs = data[0][split:]
    test_targets = data[1][split:]
    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    history = {"loss": [], "accuracy": []}

    model.train()
    for epoch in range(100):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_inputs, batch_targets in loader:
            pred = model(batch_inputs)
            loss = criterion(pred, batch_targets)

            optimizer.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()

            total_loss += loss * len(batch_inputs)
            total_correct += int(
                ((pred >= 0.5).astype(np.float64) == batch_targets).sum()
            )
            total_samples += len(batch_inputs)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        if epoch % 10 == 0 or epoch == 99:
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%"
            )

    model.eval()
    pred = model(test_inputs)
    correct = int(((pred >= 0.5).astype(np.float64) == test_targets).sum())
    test_accuracy = correct / len(test_targets) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_targets)})")
    return model, test_accuracy, history


if __name__ == "__main__":
    train()
