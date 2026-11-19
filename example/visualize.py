from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from circle import make_circle_data
from circle import train as train_circle
from matplotlib.colors import ListedColormap
from spiral import make_spiral_data
from spiral import train as train_spiral
from xor import make_xor_data
from xor import train as train_xor

FIGURES = Path(__file__).resolve().parent.parent / "figures"

BINARY_CMAP = "RdBu_r"
BINARY_COLORS = ["#3d5a80", "#c1121f"]
SPIRAL_COLORS = ["#d1495b", "#2e8b57", "#3d5a80"]
SPIRAL_BACKGROUND = ListedColormap(["#f6d3d3", "#d3ecd3", "#d3dcf0"])


def plot_loss(ax, history):
    ax.plot(history["loss"], color="#3d5a80", linewidth=1.6)
    ax.set_title("loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)


def make_grid(x_range, y_range, resolution=220):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_x, grid_y, points


def plot_binary_boundary(ax, model, inputs, targets, x_range, y_range, marker_size=18):
    grid_x, grid_y, points = make_grid(x_range, y_range)
    probs = model(points).reshape(grid_x.shape)

    ax.contourf(grid_x, grid_y, probs, levels=20, cmap=BINARY_CMAP, alpha=0.3)
    ax.contour(grid_x, grid_y, probs, levels=[0.5], colors="#2b2b2b", linewidths=1.2)
    ax.scatter(
        inputs[:, 0],
        inputs[:, 1],
        c=[BINARY_COLORS[label] for label in targets.ravel().astype(int)],
        edgecolors="white",
        linewidths=0.6,
        s=marker_size,
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_multiclass_boundary(ax, model, inputs, targets, x_range, y_range):
    grid_x, grid_y, points = make_grid(x_range, y_range)
    labels = model(points).argmax(axis=1).reshape(grid_x.shape)

    ax.contourf(
        grid_x, grid_y, labels, levels=[-0.5, 0.5, 1.5, 2.5], cmap=SPIRAL_BACKGROUND
    )
    ax.scatter(
        inputs[:, 0],
        inputs[:, 1],
        c=[SPIRAL_COLORS[label] for label in targets],
        edgecolors="white",
        linewidths=0.6,
        s=18,
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def main():
    FIGURES.mkdir(exist_ok=True)

    print("== xor ==")
    model, accuracy, history = train_xor()
    inputs, targets = make_xor_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_loss(axes[0], history)
    plot_binary_boundary(
        axes[1], model, inputs, targets, (-0.3, 1.3), (-0.3, 1.3), marker_size=70
    )
    axes[1].set_title(f"decision boundary (akurasi {accuracy:.1f}%)")
    fig.suptitle("XOR dengan MLP")
    fig.tight_layout()
    fig.savefig(FIGURES / "xor.png", dpi=110)
    plt.close(fig)

    print("\n== circle ==")
    model, accuracy, history = train_circle()
    inputs, targets = make_circle_data(500)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_loss(axes[0], history)
    plot_binary_boundary(axes[1], model, inputs, targets, (-2.2, 2.2), (-2.2, 2.2))
    axes[1].set_title(f"decision boundary (akurasi uji {accuracy:.1f}%)")
    fig.suptitle("Klasifikasi lingkaran")
    fig.tight_layout()
    fig.savefig(FIGURES / "circle.png", dpi=110)
    plt.close(fig)

    print("\n== spiral ==")
    model, accuracy, history = train_spiral()
    inputs, targets = make_spiral_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_loss(axes[0], history)
    plot_multiclass_boundary(
        axes[1], model, inputs, targets, (-1.25, 1.25), (-1.25, 1.25)
    )
    axes[1].set_title(f"decision boundary (akurasi uji {accuracy:.1f}%)")
    fig.suptitle("Klasifikasi spiral 3 kelas")
    fig.tight_layout()
    fig.savefig(FIGURES / "spiral.png", dpi=110)
    plt.close(fig)

    print(f"\nGambar disimpan di {FIGURES}")


if __name__ == "__main__":
    main()
