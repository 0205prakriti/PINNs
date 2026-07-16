""""
  ASSIGNMENT 001 — Problem 1


  Approximate the piecewise oscillatory function

        y(x) = 5 + Σ_{k=1}^{6} sin(kx),   x < 0
               cos(10x),                    x ≥ 0

  using a small neural network with tanh / ReLU activations.
  Compare results for 40 and 80 data points that are:
    (a) equi-spaced
    (b) randomly sampled from Uniform(−π, π)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ── Reproducibility 
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── Target function 
def target_function(x: torch.Tensor) -> torch.Tensor:
    """
    Piecewise function:
        y = 5 + Σ_{k=1}^6 sin(kx)   for x < 0
            cos(10x)                  for x ≥ 0
    """
    return torch.where(
        x < 0,
        5 + sum(torch.sin(k * x) for k in range(1, 7)),
        torch.cos(10 * x)
    )


# ── Network architecture
class ShallowNet(nn.Module):
    """
    One hidden layer network.
    input(1) → Linear(1→hidden_dim) → activation → Linear(hidden_dim→1)
    """
    def __init__(self, hidden_dim: int = 64, activation: str = "tanh"):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ── Training loop 
def train_model(model: nn.Module,
                x_train: torch.Tensor,
                y_train: torch.Tensor,
                epochs: int = 5000,
                lr: float = 1e-3) -> list:
    """Train with MSE loss + Adam. Returns loss log every 500 epochs."""
    optimiser = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    losses    = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimiser.step()

        if epoch % 500 == 0:
            losses.append(loss.item())

    return losses


# ── Main 
def main():
    print("=" * 60)
    print("  PROBLEM 1 — Approximating the Oscillatory Function")
    print("=" * 60)

    set_seed(42)

    # Dense test grid
    x_test = torch.linspace(-np.pi, np.pi, 1000).unsqueeze(1)
    y_test = target_function(x_test.squeeze())

    n_points_list  = [40, 80]
    sampling_list  = ["equispaced", "random"]
    activation_list = ["tanh", "relu"]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 8),
                             sharex=True, sharey=False)
    fig.suptitle("Problem 1 — tanh vs ReLU, 40 vs 80 points, "
                 "equi-spaced vs random", fontsize=14, fontweight="bold")

    col = 0
    for n_pts in n_points_list:
        for sampling in sampling_list:
            # Build training data
            
            if sampling == "equispaced":
                x_np = np.linspace(-np.pi, np.pi, n_pts)
            else:
                x_np = np.sort(np.random.uniform(-np.pi, np.pi, n_pts))

            x_tr = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
            y_tr = target_function(x_tr.squeeze()).unsqueeze(1)

            print(f"\n  n={n_pts}, sampling={sampling}")

            for row, act in enumerate(activation_list):
                set_seed(42)
                net = ShallowNet(hidden_dim=128, activation=act)
                train_model(net, x_tr, y_tr, epochs=8000, lr=1e-3)

                net.eval()
                with torch.no_grad():
                    y_pred = net(x_test).squeeze()

                rel_err = (
                    torch.sqrt(torch.mean((y_pred - y_test) ** 2)) /
                    torch.sqrt(torch.mean(y_test ** 2))
                ).item()
                print(f"    activation={act}  →  rel-L2 error = {rel_err:.4f}")

                ax = axes[row][col]
                ax.plot(x_test.numpy(), y_test.numpy(),
                        color="steelblue", lw=2, label="True f(x)")
                ax.plot(x_test.numpy(), y_pred.numpy(),
                        color="crimson", lw=1.5, linestyle="--",
                        label=f"Network (err={rel_err:.3f})")
                ax.scatter(x_np, y_tr.numpy(), s=10, color="black",
                           zorder=5, label="Train pts")
                ax.set_title(f"act={act}, n={n_pts}, {sampling}", fontsize=9)
                ax.legend(fontsize=7)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            col += 1

    plt.tight_layout()
    plt.savefig("problem1_results.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("\n  ✓ Plot saved → problem1_results.png")


if __name__ == "__main__":
    main()