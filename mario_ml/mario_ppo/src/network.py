# src/network.py

import torch
from torch import nn

from config import DEVICE


class CNNNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Initializes the CNNNetwork with specified input and output dimensions.

        Args:
            in_dim (tuple): Input dimensions (channels, height, width).
            out_dim (int): Number of outputs for the final layer.

        """
        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = DEVICE
        self.to(self.device)

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        # Compute the output size after conv layers
        conv_out_size = self._get_conv_out(in_dim)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),  # Use dynamically calculated conv_out_size
            nn.ReLU(),
            nn.Linear(512, out_dim),  # Output layer with 'out_dim' outputs
        )

    def _get_conv_out(self, shape):
        """Passes a dummy input through conv layers to get the output size."""
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, *shape
            )  # Create a dummy input with the specified shape
            output = self.conv(dummy_input)  # Pass through conv layers
            return int(output.view(1, -1).size(1))  # Flatten and get the total size

    def forward(self, state):
        """Forward pass through the CNN.

        Args:
            state (torch.Tensor): The input state (batch of images).

        Returns:
            torch.Tensor: The output of the network.

        """
        x = state.clone().detach().to(dtype=torch.float, device=self.device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)




