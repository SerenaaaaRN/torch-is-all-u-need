# TinyTorch

Implementasi mini-library deep learning (terinspirasi oleh PyTorch) untuk kebutuhan Jaringan Syaraf Tiruan feedforward. Di dalamnya ada Linear, BatchNorm, ReLU, Sigmoid, Tanh, Dropout, Sequential, MSELoss, BCELoss, SGD, Adam, dan DataLoader. Forward pass dan backpropagation ditulis manual tanpa tensor dan tanpa autograd. Proyek ini dibangun sepenuhnya dari nol hanya menggunakan Python standar tanpa dependency eksternal apa pun.

`Python 3.12+` · `zero dependencies`

## Fitur

| Kategori | Komponen |
|---|---|
| Layer | `Linear`, `BatchNorm` |
| Aktivasi | `ReLU`, `Sigmoid`, `Tanh` |
| Regularisasi | `Dropout` |
| Container | `Sequential` |
| Loss | `MSELoss`, `BCELoss` |
| Optimizer | `SGD`, `Adam` |
| Data | `DataLoader` |
| Base | `Module` |

## Instalasi

```bash
git clone <url>
cd torch-is-all-u-need
uv sync
```

Jalankan demo klasifikasi lingkaran 2D:

```bash
uv run python example/circle.py
```

## Quick Start

```python
from tinytorch import Adam, BCELoss, Linear, ReLU, Sequential, Sigmoid

model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1), Sigmoid())
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.01)

x, t = [0.5, -0.2], [1.0]

pred = model.forward(x)
loss = criterion(pred, t)

optimizer.zero_grad()
model.backward(criterion.backward())
optimizer.step()
```

`parameters()` mengembalikan referensi ke bobot dan gradien, sehingga optimizer memperbarui bobot secara in-place.

## Struktur Project

```
tinytorch/
├── module.py      # base class Module
├── nn/            # Linear, aktivasi, Dropout, BatchNorm, Sequential
├── loss.py        # MSELoss, BCELoss
├── optim.py       # SGD, Adam
└── data.py        # DataLoader
example/
└── circle.py      # demo klasifikasi lingkaran 2D
```

## Roadmap

- [ ] `Conv2D`, `MaxPool2D`, `Flatten`
- [ ] Training per-batch dengan gradien dirata-rata
- [ ] Unit test
- [ ] Contoh MNIST MLP

## Lisensi

MIT. File `LICENSE` belum ditambahkan.
