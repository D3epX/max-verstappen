"""Construct the shared encoder and separate actor/critic heads."""
import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from .config import PPOConfig
from .models.cnn.network import BasicCNN
from .models.impala_cnn.network import ImpalaCNN


def policy_kwargs(config: PPOConfig):
    """Layer layout and distribution settings, compatible with notebook tensors."""
    encoder = {"basic_cnn": BasicCNN, "impala_cnn": ImpalaCNN}[config.architecture]
    return {
        "features_extractor_class": encoder,
        "features_extractor_kwargs": {"features_dim": config.features_dim},
        "net_arch": {"pi": [64], "vf": [64]},
        "share_features_extractor": True,
        "normalize_images": True,
        "log_std_init": config.log_std_init,
    }


def make_policy(config: PPOConfig, device="cpu"):
    """Build an inference policy without creating a simulator or PPO trainer.

    Actor: encoder -> 64 Tanh units -> three Gaussian means.
    Critic: same encoder -> separate 64 Tanh units -> one value.
    gSDE scales belong to the action distribution; deterministic inference uses
    the mean, clips to the legal action bounds and adds no exploration noise.
    """
    observations = gym.spaces.Box(0, 255, shape=(config.frame_stack, config.image_size, config.image_size), dtype=np.uint8)
    actions = gym.spaces.Box(np.array([-1, 0, 0], dtype=np.float32), np.ones(3, dtype=np.float32))
    # Every tensor will be replaced when loading, so skip orthogonal initialization.
    policy = ActorCriticCnnPolicy(
        observations, actions, lr_schedule=lambda _: config.learning_rate,
        use_sde=config.use_sde, ortho_init=False, **policy_kwargs(config),
    ).to(device)
    policy.set_training_mode(False)
    return policy
