"""The notebook-compatible three-convolution image encoder.

SB3 normalizes uint8 pixels before forward(); dividing by 255 here again would
make observations too small. Layer names intentionally match saved checkpoints.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BasicCNN(BaseFeaturesExtractor):
    """Three convolutional layers followed by a 256-number feature vector."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            flattened_size = self.convolutions(dummy).shape[1]
        self.projection = nn.Sequential(
            nn.Linear(flattened_size, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        # SB3 already converts uint8 pixels to float32 / 255 before this call.
        return self.projection(self.convolutions(observations))
