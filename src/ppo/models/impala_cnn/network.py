"""Residual image encoder used with PPO, not the distributed IMPALA algorithm.

Three stages downsample 84 -> 42 -> 21 -> 11. Each stage has two residual
blocks; shortcut additions preserve a direct path for features and gradients.
Layer names match the self-contained Colab notebook checkpoints.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    """Learn a change F(x), then return x + F(x). Channels stay the same."""
    def __init__(self, channels):
        super().__init__()
        self.change = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.change(x)

class ImpalaStage(nn.Module):
    """Change channel count, downsample, then apply two residual blocks."""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(output_channels), ResidualBlock(output_channels),
        )

    def forward(self, x):
        return self.layers(x)

class ImpalaCNN(BaseFeaturesExtractor):
    """Small IMPALA-style image encoder, used WITH PPO, not IMPALA learning."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.stages = nn.Sequential(
            ImpalaStage(channels, 16),
            ImpalaStage(16, 32),
            ImpalaStage(32, 32),
            nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            flattened_size = self.stages(dummy).shape[1]
        self.projection = nn.Sequential(
            nn.Linear(flattened_size, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        # The image has already been normalized by SB3, exactly once.
        return self.projection(self.stages(observations))
