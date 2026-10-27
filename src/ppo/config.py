"""Settings for the local version of our CarRacing notebook experiments.

This file contains settings only. Importing it does not create a game, allocate
a GPU, load weights, or start training. PPO and the CNN will use these settings
when the local trainer creates the model.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import math


# These are road identifiers, not difficulty levels or neural-network seeds.
# Keep validation and final-test roads separate from the roads used to learn.
# Public-track fine-tuning is a later experiment; it is NOT enabled here.
TRAINING_SEEDS = tuple(range(1000, 1100))
VALIDATION_SEEDS = tuple(range(10000, 10020))
TEST_SEEDS = tuple(range(20000, 20020))
PUBLIC_SEEDS = (202, 42, 203)


@dataclass(frozen=True)
class PPOConfig:
    """The model and learning settings shared by the two gSDE notebooks.

    A dataclass gives us a named container for settings. frozen=True prevents
    accidental reassignment during a run. Create a NEW configuration when
    intentionally changing an experiment, instead of changing it halfway.
    """

    # This chooses the image encoder, not a different RL algorithm.
    architecture: Literal["basic_cnn", "impala_cnn"] = "basic_cnn"
    # Controls initial weights and sampling randomness; different from a road ID.
    run_seed: int = 7
    training_seeds: tuple[int, ...] = TRAINING_SEEDS

    # Raw RGB images are 96x96. The policy sees four grayscale 84x84 images.
    image_size: int = 84
    frame_stack: int = 4
    # One decision normally advances two simulator steps and sums their rewards.
    action_repeat: int = 2

    # Each of two environments contributes 512 decisions to one PPO rollout.
    n_envs: int = 2
    n_steps: int = 512
    # 1024 transitions / 128 = eight minibatches per full update pass.
    batch_size: int = 128
    # Up to five shuffled passes over a rollout; the KL guard can stop earlier.
    n_epochs: int = 5

    learning_rate: float = 3e-4
    # Discount and GAE weighting apply per decision, not per raw physics step.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # PPO probability-ratio clipping; NOT clipping steering to legal bounds.
    clip_range: float = 0.2
    # No EXTRA randomness bonus. The training policy still samples actions.
    ent_coef: float = 0.0
    # Multiplies critic loss in the combined loss that updates shared features.
    vf_coef: float = 0.5

    # Match our recovery: feature-dependent noise, resampled every four decisions.
    use_sde: bool = True
    sde_sample_freq: int = 4
    # exp(-2) initializes noise-WEIGHT scales, not a fixed action standard deviation.
    log_std_init: float = -2.0
    # Limits gradient norm; unrelated to vf_coef despite the same numeric value.
    max_grad_norm: float = 0.5
    # SB3 can stop update passes when approximate KL exceeds 1.5 * this value.
    target_kl: float = 0.03

    # The CNN returns this many features to the actor and critic heads.
    features_dim: int = 256

    def __post_init__(self) -> None:
        """Reject common configuration mistakes before a long run starts."""
        if self.architecture not in ("basic_cnn", "impala_cnn"):
            raise ValueError("architecture must be 'basic_cnn' or 'impala_cnn'")
        for name in (
            "image_size", "frame_stack", "action_repeat", "n_envs", "n_steps",
            "batch_size", "n_epochs", "features_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2 or self.rollout_size % self.batch_size:
            raise ValueError("batch_size must be at least 2 and divide n_envs * n_steps")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("learning_rate", "clip_range", "max_grad_norm", "target_kl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ent_coef", "vf_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.log_std_init):
            raise ValueError("log_std_init must be finite")
        if self.sde_sample_freq != -1 and self.sde_sample_freq < 1:
            raise ValueError("sde_sample_freq must be -1 or a positive integer")
        if not 0 <= self.run_seed < 2**32:
            raise ValueError("run_seed must be between 0 and 2**32 - 1")
        if not isinstance(self.training_seeds, tuple) or not self.training_seeds:
            raise ValueError("training_seeds must be a nonempty tuple")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must not contain duplicates")
        for seed in self.training_seeds:
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
                raise ValueError("each training seed must be an integer in [0, 2**32)")
        if set(self.training_seeds) & set(VALIDATION_SEEDS + TEST_SEEDS + PUBLIC_SEEDS):
            raise ValueError("this baseline keeps validation, test and public seeds out of training")

    @property
    def rollout_size(self) -> int:
        """Total transitions collected before one PPO update phase: 1024."""
        return self.n_envs * self.n_steps

    def to_dict(self) -> dict:
        """Return the notebook-compatible CONFIG format for JSON/checkpoints."""
        settings = asdict(self)
        # JSON stores arrays, not Python tuples. Match notebook metadata exactly.
        settings["training_seeds"] = list(self.training_seeds)
        return settings


@dataclass(frozen=True)
class RunConfig:
    """Execution choices, kept separate from the model's learning settings.

    Increasing the target or changing CPU/GPU should not be mistaken for a new
    architecture or new PPO hyperparameters when resuming the same experiment.
    """

    # Cumulative across environments, not 500k per episode or 500k additional.
    target_total_timesteps: int = 500_000
    # Race length in RAW simulator steps: 3000 / 50 = 60 simulator seconds.
    max_episode_steps: int = 3000
    save_every: int = 25_000
    evaluate_every: int = 100_000
    # Passed to SB3 later: auto chooses an available device; it creates no GPU.
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Avoid the Rich/IPython recursion observed in our Colab progress display.
    progress_bar: bool = False
    # The trainer will make root / architecture / experiment_name, not this file.
    output_root: Path = Path("racing_outputs")
    experiment_name: str = "local_gsde_01"

    def __post_init__(self) -> None:
        """Catch invalid budgets, device names and unsafe folder components."""
        for name in ("target_total_timesteps", "max_episode_steps", "save_every", "evaluate_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if not isinstance(self.output_root, Path):
            raise ValueError("output_root must be a pathlib.Path")
        if not self.experiment_name or self.experiment_name in (".", "..") or any(
            character in self.experiment_name for character in ("/", "\\")
        ):
            raise ValueError("experiment_name must be one nonempty folder name")
