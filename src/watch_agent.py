"""Watch a trained checkpoint from this project's CarRacing notebooks.

Examples:
    python src/watch_agent.py --model ~/Downloads/best_validation.zip --seed 10000
    python src/watch_agent.py --model ~/Downloads/latest.zip --video preview.mp4

Inference only: no training or optimizer updates. The original and gSDE policies
are supported. Tensor-only loading avoids unpickling a notebook-defined Python
class saved under a different Python minor version.
"""

import argparse
import io
import json
import os
from pathlib import Path
import zipfile

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

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


Encoder = BasicCNN

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


Encoder = ImpalaCNN

class TrackSampler(gym.Wrapper):
    """Choose tracks from one explicit pool; the policy never receives the seed."""
    def __init__(self, env, seeds, random_order=True, sampler_seed=7):
        super().__init__(env)
        self.seeds = tuple(int(seed) for seed in seeds)
        if not self.seeds:
            raise ValueError("The track pool cannot be empty")
        self.random_order = random_order
        self.rng = np.random.default_rng(sampler_seed)
        self.index = 0
        self.track_seed = None

    def reset(self, *, seed=None, options=None):
        # Here seed controls the SAMPLER. The chosen track seed is passed on
        # to CarRacing. SB3 seeds each worker once before its first reset.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.index = 0
        if self.random_order:
            self.track_seed = int(self.rng.choice(self.seeds))
        else:
            self.track_seed = self.seeds[self.index % len(self.seeds)]
            self.index += 1
        observation, info = self.env.reset(seed=self.track_seed, options=options)
        return observation, {**info, "track_seed": self.track_seed}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, {
            **info, "track_seed": self.track_seed,
        }


class ActionRepeat(gym.Wrapper):
    """Repeat a decision, sum rewards, and count actual simulator steps."""
    def __init__(self, env, repeat=2):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be positive")
        self.repeat = repeat
        self.raw_steps = 0

    def reset(self, **kwargs):
        self.raw_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.raw_steps += 1
            total_reward += float(reward)
            # Never continue stepping after either kind of episode ending.
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, {
            **info, "raw_steps": self.raw_steps,
        }


class GrayResize(gym.ObservationWrapper):
    """RGB -> resized grayscale; keep uint8 until SB3 normalizes it once."""
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8,
        )

    def observation(self, observation):
        image = Image.fromarray(observation).convert("L")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)[..., None].copy()


def make_single_env(seeds, random_order, sampler_seed, config, render_mode=None):
    # TimeLimit is INSIDE action repetition: 3000 means raw simulator steps.
    env = gym.make(
        "CarRacing-v3", continuous=True, max_episode_steps=config.get("max_episode_steps", 3000),
        render_mode=render_mode,
    )
    env = TrackSampler(env, seeds, random_order, sampler_seed)
    env = ActionRepeat(env, repeat=config["action_repeat"])
    env = GrayResize(env, size=config["image_size"])
    return Monitor(env)


def load_policy(checkpoint_path):
    """Reconstruct our reviewed policy layout and strictly load its tensors."""
    with zipfile.ZipFile(checkpoint_path) as archive:
        metadata = json.loads(archive.read("data"))
        state = torch.load(
            io.BytesIO(archive.read("policy.pth")),
            map_location="cpu", weights_only=True,
        )
    config = metadata.get("experiment_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint lacks this project's experiment configuration")
    encoders = {"basic_cnn": BasicCNN, "impala_cnn": ImpalaCNN}
    architecture = config.get("architecture")
    if architecture not in encoders:
        raise ValueError(f"Unsupported architecture: {architecture}")
    kwargs = metadata.get("policy_kwargs", {})
    if kwargs.get("net_arch") != {"pi": [64], "vf": [64]}:
        raise ValueError("Unexpected actor/critic layout; use its matching source")
    if not kwargs.get("normalize_images", True):
        raise ValueError("Expected the notebooks' image normalization")
    size = int(config["image_size"])
    observations = gym.spaces.Box(
        0, 255, shape=(int(config["frame_stack"]), size, size), dtype=np.uint8,
    )
    actions = gym.spaces.Box(
        np.array([-1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1], dtype=np.float32),
    )
    policy = ActorCriticCnnPolicy(
        observations, actions, lr_schedule=lambda _: float(config["learning_rate"]),
        features_extractor_class=encoders[architecture],
        features_extractor_kwargs={"features_dim": int(config["features_dim"])},
        net_arch={"pi": [64], "vf": [64]},
        share_features_extractor=True, normalize_images=True,
        use_sde=bool(metadata.get("use_sde", False)),
        log_std_init=float(config.get("log_std_init", 0.0)),
        # Every parameter is replaced below; orthogonal initialization is wasted work.
        ortho_init=False,
    )
    policy.load_state_dict(state, strict=True)
    policy.set_training_mode(False)
    if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Policy contains non-finite weights")
    return policy, config, int(metadata["num_timesteps"])


def watch(checkpoint_path, seed=10000, video_path=None, no_window=False, max_raw_steps=3000):
    """Run one deterministic episode with the same preprocessing as training."""
    policy, config, transitions = load_policy(checkpoint_path)
    config = dict(config, max_episode_steps=max_raw_steps)
    render_mode = "rgb_array" if video_path else (None if no_window else "human")
    if render_mode != "human":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = DummyVecEnv([
        lambda: make_single_env((seed,), False, config["run_seed"], config, render_mode)
    ])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=config["frame_stack"], channels_order="first")
    writer = None
    total_reward = 0.0
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {config['architecture']}; gSDE: {policy.use_sde}")
    print(f"Training transitions: {transitions:,}; track: {seed}")
    try:
        observation = env.reset()
        if video_path:
            import imageio.v2 as imageio
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=50 / config["action_repeat"])
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(action)
            total_reward += float(rewards[0])
            if writer is not None and not dones[0]:
                frame = Image.fromarray(env.envs[0].render()).resize((480, 320))
                writer.append_data(np.asarray(frame))
            if dones[0]:
                info = infos[0]
                if not info.get("TimeLimit.truncated", False) and "lap_finished" not in info:
                    raise RuntimeError("Missing lap completion metadata; check Gymnasium version")
                finished = bool(info.get("lap_finished", False))
                raw_steps = int(info["raw_steps"])
                result = {
                    "checkpoint_transitions": transitions, "track_seed": seed,
                    "finished": finished, "raw_steps": raw_steps,
                    "configured_raw_step_limit": max_raw_steps,
                    "lap_seconds": raw_steps / 50 if finished else None,
                    "penalized_seconds": raw_steps / 50 if finished else 70.0,
                    "reward": total_reward,
                }
                print(json.dumps(result, indent=2))
                return result
    finally:
        if writer is not None:
            writer.close()
            print(f"Video saved: {video_path}")
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Notebook's latest.zip or best_validation.zip")
    parser.add_argument("--seed", type=int, default=10000, help="Track seed to view")
    parser.add_argument("--video", type=Path, help="Record MP4 instead of opening a window")
    parser.add_argument("--no-window", action="store_true", help="Evaluate without a window or video")
    parser.add_argument("--max-raw-steps", type=int, default=3000, help="Debug limit; leave 3000 for challenge-style evaluation")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Checkpoint does not exist: {args.model}")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if args.max_raw_steps < 1:
        parser.error("max-raw-steps must be positive")
    torch.set_num_threads(2)
    watch(args.model, args.seed, args.video, args.no_window, args.max_raw_steps)


if __name__ == "__main__":
    main()
