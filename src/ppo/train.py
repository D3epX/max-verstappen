"""Train or resume local continuous-action PPO: python -m src.ppo.train --help."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import time
import torch
from stable_baselines3.common.logger import configure
from .config import PPOConfig, RunConfig
from .environment import build_vec_env
from .checkpoint import read_checkpoint, config_from_metadata, resume_model, save_checkpoint
from .model import make_model
from .callbacks import RacingCallback


def train(config: PPOConfig, run: RunConfig, resume=None):
    """Continue to a cumulative budget; always attempt to save on interruption.

    Choosing a smaller total chunk does not change PPO's rollout or minibatches.
    Saved optimizer/counter resume is practical continuation, not simulator replay.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(2)
    if run.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu or auto")
    run_dir = run.output_root / config.architecture / run.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config.to_dict():
        raise ValueError("This folder has different model settings; choose a new experiment name")
    if (run_dir / "latest.zip").exists() and resume is None:
        raise FileExistsError("A checkpoint exists; pass --resume or choose a new experiment name")
    run_path = run_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["max_episode_steps"] != run.max_episode_steps:
        raise ValueError("Race limit changed; use a new folder for a different environment")
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    run_path.write_text(json.dumps({**asdict(run), "output_root": str(run.output_root),
                                   "resume_from": str(resume) if resume else None}, indent=2))
    (run_dir / "requirements_runtime.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    env = build_vec_env(config, max_episode_steps=run.max_episode_steps)
    model = None
    try:
        model = resume_model(resume, env, run.device) if resume else make_model(config, env, run.device)
        if model.experiment_config != config.to_dict():
            raise ValueError("Resume checkpoint settings do not match requested configuration")
        model.set_logger(configure(str(run_dir / "logs"), ["stdout", "csv", "tensorboard"]))
        print("Run folder:", run_dir)
        print("Actual parameter device:", next(model.policy.parameters()).device)
        print("Starting transitions:", model.num_timesteps)
        remaining = max(0, run.target_total_timesteps - model.num_timesteps)
        if remaining:
            start = time.perf_counter()
            start_steps = model.num_timesteps
            try:
                model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                            callback=RacingCallback(run_dir, config, run),
                            progress_bar=run.progress_bar)
            except KeyboardInterrupt:
                print("Interrupted; preserving the current policy and optimizer.")
            finally:
                saved = save_checkpoint(model, run_dir / "latest.zip")
                elapsed = time.perf_counter() - start
                print("Checkpoint:", saved)
                print(f"Collected {model.num_timesteps - start_steps:,} transitions in {elapsed / 60:.1f} minutes")
        else:
            print("Cumulative target already reached; no new training performed.")
        return run_dir
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("basic_cnn", "impala_cnn"))
    parser.add_argument("--resume", type=Path, help="SB3 ZIP; preserves saved optimizer and counter")
    parser.add_argument("--target", type=int, default=500_000, help="Cumulative transitions, not additional")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("racing_outputs"))
    parser.add_argument("--experiment", default="local_gsde_01")
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--evaluate-every", type=int, default=100_000)
    parser.add_argument("--max-raw-steps", type=int, default=3000, help="Shorter is debug-only")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--run-seed", type=int, default=None)
    args = parser.parse_args()
    overrides = {key: getattr(args, key) for key in
                 ("architecture", "n_steps", "batch_size", "n_epochs", "learning_rate", "run_seed")
                 if getattr(args, key) is not None}
    if args.resume:
        metadata, _ = read_checkpoint(args.resume)
        if metadata.get("training_sampling"):
            parser.error("This checkpoint used a different training sampler and cannot be resumed by this uniform-sampling trainer.")
        config = config_from_metadata(metadata)
        if any(getattr(config, key) != value for key, value in overrides.items()):
            parser.error("Resume keeps the checkpoint settings. Use a fresh experiment for hyperparameter changes.")
    else:
        config = PPOConfig(**overrides)
    run = RunConfig(target_total_timesteps=args.target, device=args.device,
                    output_root=args.output_root, experiment_name=args.experiment,
                    max_episode_steps=args.max_raw_steps, save_every=args.save_every,
                    evaluate_every=args.evaluate_every)
    train(config, run, resume=args.resume)


if __name__ == "__main__":
    main()
