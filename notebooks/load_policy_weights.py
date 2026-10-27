"""Load this project's policy for inference without unpickling notebook code.

SB3's ordinary PPO.load remains the training-resume method in a matching Python
runtime. This helper instead reconstructs the known architecture and loads only
its tensors, useful when Colab and the inspection machine use different Python
minor versions. It does NOT restore optimizer/environment state or resume training.
"""

import io
import json
import zipfile
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticCnnPolicy


def load_policy_weights(checkpoint_path, encoder_class, device="cpu"):
    """Read native experiment metadata and tensors from our baseline SB3 zip.

    Supply BasicCNN or ImpalaCNN from the matching notebook. Never deserialize
    the embedded feature-extractor class: reconstruct it from the reviewed local
    source. Strict state loading catches differences in parameter shapes/names.
    """
    with zipfile.ZipFile(checkpoint_path) as archive:
        metadata = json.loads(archive.read("data"))
        state = torch.load(
            io.BytesIO(archive.read("policy.pth")),
            map_location=device,
            weights_only=True,
        )
    config = metadata.get("experiment_config")
    if not isinstance(config, dict):
        raise ValueError("Missing this project's experiment configuration")
    if metadata.get("use_sde", False):
        raise ValueError("This helper supports the original diagonal-Gaussian notebooks only")
    kwargs = metadata.get("policy_kwargs", {})
    if kwargs.get("net_arch") != {"pi": [64], "vf": [64]}:
        raise ValueError("Unexpected actor/critic layout; reconstruct it explicitly")
    if not kwargs.get("normalize_images", True):
        raise ValueError("Expected SB3's single image-normalization step")
    image_size = int(config["image_size"])
    observation_space = gym.spaces.Box(
        0, 255, shape=(int(config["frame_stack"]), image_size, image_size),
        dtype=np.uint8,
    )
    action_space = gym.spaces.Box(
        np.array([-1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1], dtype=np.float32),
    )
    policy = ActorCriticCnnPolicy(
        observation_space, action_space,
        lr_schedule=lambda _: float(config["learning_rate"]),
        features_extractor_class=encoder_class,
        features_extractor_kwargs={"features_dim": int(config["features_dim"])},
        net_arch={"pi": [64], "vf": [64]},
        share_features_extractor=True, normalize_images=True,
        # Initialization is irrelevant because every parameter is replaced.
        ortho_init=False,
    ).to(device)
    policy.load_state_dict(state, strict=True)
    policy.set_training_mode(False)
    if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite policy tensors")
    return SimpleNamespace(
        policy=policy,
        predict=policy.predict,
        device=policy.device,
        observation_space=observation_space,
        action_space=action_space,
        num_timesteps=int(metadata["num_timesteps"]),
        experiment_config=config,
        use_sde=False,
    )
