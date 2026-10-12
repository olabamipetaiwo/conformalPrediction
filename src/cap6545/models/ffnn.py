"""Feed-forward network for binding-affinity regression on ECIF features.

Reproduces the Rayka & Naghavi 2025 architecture: 790 -> 900 -> 300 -> 1 with ELU
activations and dropout after each hidden layer. The architecture is configurable
so the deep-ensemble (same net, different seeds) and MC-dropout (same net, dropout
kept on at inference) uncertainty variants reuse this one backbone.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class FFNN(nn.Module):
    """Configurable MLP regressor producing a single scalar (predicted pK)."""

    def __init__(
        self,
        in_features: int = 790,
        hidden_units: Sequence[int] = (900, 300),
        dropout: float = 0.4,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}"
            )
        act = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev = in_features
        for units in hidden_units:
            layers += [nn.Linear(prev, units), act(), nn.Dropout(dropout)]
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _enable_dropout(model: nn.Module) -> None:
    """Put only dropout layers into train mode, leaving the rest in eval."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: FFNN, x: torch.Tensor, samples: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) over ``samples`` stochastic forward passes with dropout on.

    The model is set to eval first (freezing e.g. batchnorm), then dropout layers
    are reactivated, so uncertainty comes only from dropout.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    model.eval()
    _enable_dropout(model)
    preds = torch.stack([model(x) for _ in range(samples)], dim=0)
    return preds.mean(dim=0), preds.std(dim=0)
