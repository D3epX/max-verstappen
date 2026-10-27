"""Atomic checkpoints and tensor-only portability between notebook/local Python.

Notebook classes saved by cloudpickle can fail across Python minor versions.
We read native JSON metadata, recreate OUR known layout and load tensor states.
We never deserialize the checkpoint's embedded Python class or lambda objects.
"""
from dataclasses import fields
import io
import json
import os
from pathlib import Path
import zipfile
import torch
from .config import PPOConfig
from .policy import make_policy


def read_checkpoint(path):
    """Read an SB3 ZIP or our exported bundle without unpickling notebook classes."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            # A torch .pth is also a ZIP internally: recognize the SB3 file layout.
            if "data" in archive.namelist() and "policy.pth" in archive.namelist():
                metadata = json.loads(archive.read("data"))
                state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
                return metadata, state
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or not {"config", "num_timesteps", "policy_state_dict"}.issubset(bundle):
        raise ValueError("Expected an SB3 notebook ZIP or project policy-weight bundle")
    if bundle.get("format", "carracing-policy-v1") != "carracing-policy-v1":
        raise ValueError("Unsupported export format")
    return {"experiment_config": bundle["config"], "num_timesteps": bundle["num_timesteps"],
            "use_sde": bundle["config"].get("use_sde", False),
            "training_sampling": bundle.get("training_sampling"),
            "finetune_parent": bundle.get("finetune_parent")}, bundle["policy_state_dict"]


def config_from_metadata(metadata):
    """Normalize legacy notebook settings and reject unsupported policy layouts."""
    raw = metadata.get("experiment_config")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint lacks this project's experiment_config")
    settings = dict(raw)
    settings.setdefault("use_sde", bool(metadata.get("use_sde", False)))
    settings.setdefault("sde_sample_freq", int(metadata.get("sde_sample_freq", -1)))
    settings.setdefault("log_std_init", 0.0)
    settings["training_seeds"] = tuple(settings["training_seeds"])
    allowed = {field.name for field in fields(PPOConfig)}
    if set(settings) - allowed:
        raise ValueError(f"Unsupported checkpoint settings: {set(settings) - allowed}")
    kwargs = metadata.get("policy_kwargs", {})
    if kwargs and (kwargs.get("net_arch") != {"pi": [64], "vf": [64]}
                   or not kwargs.get("normalize_images", True)
                   or not kwargs.get("share_features_extractor", True)
                   or kwargs.get("squash_output", False)):
        raise ValueError("Checkpoint does not use this project's shared normalized CNN/head layout")
    return PPOConfig(**settings)


def load_policy(path, device="cpu"):
    """Return deterministic inference policy, typed settings and training counter."""
    metadata, state = read_checkpoint(path)
    config = config_from_metadata(metadata)
    policy = make_policy(config, device=device)
    policy.load_state_dict(state, strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in policy.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite weights")
    return policy, config, int(metadata["num_timesteps"])


def resume_model(path, env, device="auto", log_dir=None):
    """Restore model AND Adam state, deliberately starting fresh simulator state.

    This supports our constant-learning-rate notebook recipe. It does not promise
    bit-identical RNG continuation, or restore partial rollout samples/log history.
    """
    from .model import make_model
    metadata, state = read_checkpoint(path)
    config = config_from_metadata(metadata)
    model = make_model(config, env, device, log_dir)
    model.policy.load_state_dict(state, strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in model.policy.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite weights")
    with zipfile.ZipFile(path) as archive:
        if "policy.optimizer.pth" not in archive.namelist():
            raise ValueError("Training resume requires an SB3 ZIP with optimizer state, not weights only")
        optimizer = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")), map_location=model.device, weights_only=True)
    model.policy.optimizer.load_state_dict(optimizer)
    model.num_timesteps = int(metadata["num_timesteps"])
    model._n_updates = int(metadata.get("_n_updates", 0))
    model._last_obs = None  # learn() must reset new environments before collection.
    return model


def save_checkpoint(model, path):
    """Write in the destination directory and atomically replace a completed ZIP."""
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("Use a .zip path for resumable training checkpoints")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.zip")
    model.save(str(temporary))
    os.replace(temporary, path)
    return path
