import torch
from torch import nn

from interfaces import Params


def build_torch_mlp(layer_sizes: list[int], params: Params) -> nn.Sequential:
    """float64 nn.Sequential with the same architecture as MLP(layer_sizes), loaded with our params"""
    layers = []
    for k in range(1, len(layer_sizes)):
        layers.append(nn.Linear(layer_sizes[k-1], layer_sizes[k]))
        if k < len(layer_sizes) - 1:
            layers.append(nn.ReLU())
    torch_model = nn.Sequential(*layers).double()

    # torch stores weight as (out, in) and computes x @ weight.T + bias, so our W (in, out) goes in transposed
    with torch.no_grad():
        for k in range(1, len(layer_sizes)):
            layer = torch_model[2 * (k-1)]
            layer.weight.copy_(torch.from_numpy(params[f"W{k}"].T))
            layer.bias.copy_(torch.from_numpy(params[f"b{k}"]))
    return torch_model
