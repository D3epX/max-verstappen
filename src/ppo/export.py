"""Export selected weights and verify deterministic raw-image inference."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from .checkpoint import read_checkpoint, config_from_metadata, load_policy
from .driver import load_driver
from .environment import build_vec_env


def export_weights(checkpoint, output):
    """Create a tensor/metadata bundle, not a pickled notebook model object."""
    checkpoint, output = Path(checkpoint), Path(output)
    if checkpoint.resolve() == output.resolve() or output.suffix not in (".pth", ".pt"):
        raise ValueError("Use a separate .pth/.pt output path")
    metadata, state = read_checkpoint(checkpoint)
    config = config_from_metadata(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "carracing-policy-v1", "config": config.to_dict(),
                "num_timesteps": int(metadata["num_timesteps"]),
                "policy_state_dict": {k: v.detach().cpu() for k, v in state.items()},
                "sb3_version": sb3.__version__, "gymnasium_version": gym.__version__,
                "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, output)
    # Strict loading catches layout/parameter mismatches before submission.
    load_driver(output)
    return output


def verify_export(checkpoint, output, seed=10000, max_raw_steps=3000):
    """Compare adapter actions against the original policy on matching raw frames.

    Two seeded simulators let us compare our per-raw-step adapter with the actual
    vector preprocessing/repeat pipeline. Check reset padding, action bounds,
    reward sums, completion and step counts, not just a synthetic zero image.
    Any divergence fails verification rather than being silently accepted.
    """
    original, config, steps = load_policy(checkpoint)
    driver = load_driver(output)
    raw_env = gym.make("CarRacing-v3", continuous=True, max_episode_steps=max_raw_steps)
    vector = build_vec_env(config, (seed,), n_envs=1, random_order=False,
                           max_episode_steps=max_raw_steps)
    try:
        raw_obs, _ = raw_env.reset(seed=seed)
        stacked = vector.reset()
        driver.reset()
        raw_steps = 0
        while raw_steps < max_raw_steps:
            expected, _ = original.predict(stacked, deterministic=True)
            actual = driver.act(raw_obs)
            np.testing.assert_allclose(actual, expected[0], rtol=1e-5, atol=1e-6)
            raw_reward = 0.0
            for repeat in range(config.action_repeat):
                if repeat:
                    np.testing.assert_allclose(driver.act(raw_obs), actual)
                raw_obs, reward, terminated, truncated, raw_info = raw_env.step(actual)
                raw_steps += 1
                raw_reward += reward
                if terminated or truncated:
                    break
            stacked, rewards, dones, infos = vector.step(expected)
            if raw_steps != int(infos[0]["raw_steps"]):
                raise AssertionError("Raw step counters differ")
            np.testing.assert_allclose(raw_reward, rewards[0], rtol=1e-5, atol=1e-5)
            if bool(dones[0]) != bool(terminated or truncated):
                raise AssertionError("Adapter and reference disagree on episode ending")
            if dones[0]:
                if bool(raw_info.get("lap_finished", False)) != bool(infos[0].get("lap_finished", False)):
                    raise AssertionError("Completion flags differ")
                break
        return {"verified_raw_steps": raw_steps, "checkpoint_transitions": steps,
                "seed": seed, "finished": bool(raw_info.get("lap_finished", False)),
                "configured_raw_step_limit": max_raw_steps}
    finally:
        raw_env.close()
        vector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--verify-steps", type=int, default=3000)
    args = parser.parse_args()
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(2)
    if args.verify_steps < 1:
        parser.error("verify-steps must be positive")
    path = export_weights(args.checkpoint, args.output)
    print("Exported:", path)
    if args.verify:
        print("Verification:", verify_export(args.checkpoint, path, args.seed, args.verify_steps))


if __name__ == "__main__":
    main()
