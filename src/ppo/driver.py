"""Image-only deterministic inference on raw CarRacing RGB observations.

Organizer-facing interface: driver = load_driver(weights); driver.reset();
action = driver.act(rgb_observation) on EVERY simulator step. Do not externally
repeat actions again: this adapter already matches the notebook's repeat timing.
"""
from collections import deque
from torch import nn
import numpy as np
from PIL import Image
from .checkpoint import load_policy


class Driver(nn.Module):
    """Keep frame history and repeat state outside the learned policy weights."""
    def __init__(self, policy, config, transitions):
        super().__init__()
        self.policy, self.config, self.num_timesteps = policy, config, transitions
        self.reset()

    def reset(self):
        """Call at every env.reset(), even when using the same track seed again."""
        size = self.config.image_size
        self.frames = deque((np.zeros((size, size), dtype=np.uint8)
                             for _ in range(self.config.frame_stack)), maxlen=self.config.frame_stack)
        self.raw_steps = 0
        self.last_action = np.zeros(3, dtype=np.float32)

    def eval(self):
        """Match the familiar PyTorch eval convention without enabling exploration."""
        return super().eval()

    def act(self, observation):
        """Return [steering, gas, brake]; never inspect simulator internal state."""
        observation = np.asarray(observation)
        if observation.shape != (96, 96, 3) or observation.dtype != np.uint8:
            raise ValueError("Expected one raw (96,96,3) uint8 RGB observation")
        if self.raw_steps % self.config.action_repeat == 0:
            image = Image.fromarray(observation).convert("L").resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR)
            self.frames.append(np.asarray(image, dtype=np.uint8).copy())
            stacked = np.stack(self.frames)
            self.last_action, _ = self.policy.predict(stacked, deterministic=True)
            self.last_action = np.asarray(self.last_action, dtype=np.float32)
        self.raw_steps += 1
        return self.last_action.copy()

    def forward(self, observation):
        return self.act(observation)


def load_driver(weights_path, device="cpu"):
    """Load our tensor-only .pth or original notebook ZIP; no internet needed."""
    return Driver(*load_policy(weights_path, device=device)).eval()
