import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Network(nn.Module):
    def __init__(self, num_of_features: int, num_of_classes: int, hidden_units: List[int], drop: float = 0.5):
        super().__init__()

        self.num_of_features = num_of_features
        self.hidden_units = hidden_units
        self.num_of_classes = num_of_classes

        # when hidden_units is empty, the network is a simple linear classifier
        if len(hidden_units) == 0:
            self.hidden_layers = None
            self.output = nn.Linear(num_of_features, num_of_classes)
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(num_of_features, hidden_units[0])])
            layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            self.output = nn.Linear(hidden_units[-1], num_of_classes)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_layers:
            for each in self.hidden_layers:
                x = F.relu(each(x))
                x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)
