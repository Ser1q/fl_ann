import torch
from torch import nn
import torch.nn.functional as F


class SimpleANN(nn.Module):
    """
    A simple fully-connected neural network for classification.
    Used as a baseline model for centralized and federated learning.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten
        x = x.view(-1, self.input_dim)

        # Hidden layer + non-linearity
        x = F.relu(self.fc1(x))

        # Output layer (logits)
        x = self.fc2(x)
        return x
