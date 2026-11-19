import numpy as np

from tinytorch import Adam, BCELoss, Linear, ReLU, Sequential, Sigmoid


def make_xor_data():
    inputs = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
    )
    targets = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    return inputs, targets


def train():
    np.random.seed(42)
    inputs, targets = make_xor_data()

    model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1), Sigmoid())
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    history = {"loss": []}

    model.train()
    for epoch in range(2000):
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        model.backward(criterion.backward())
        optimizer.step()

        history["loss"].append(loss)
        if epoch % 200 == 0 or epoch == 1999:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    model.eval()
    predictions = model(inputs)

    print("\nInput        Target  Prediction")
    for x, target, prediction in zip(inputs, targets, predictions):
        print(
            f"{int(x[0])} XOR {int(x[1])}  ->  {int(target[0])}       {prediction[0]:.4f}"
        )

    accuracy = float(((predictions >= 0.5).astype(np.float64) == targets).mean()) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    return model, accuracy, history


if __name__ == "__main__":
    train()
