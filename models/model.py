import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(
        self,
        hidden_units=128,
        dropout=0.0
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_units // 2, 10)
        )

    def forward(self, x):
        return self.network(x)
