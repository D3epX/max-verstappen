"""Deterministic full races, shared-track comparisons and checkpoint selection."""
import argparse
import hashlib
import importlib.metadata
import sys
import json
from pathlib import Path
import numpy as np
from .checkpoint import load_policy
from .config import PPOConfig, TRAINING_SEEDS, VALIDATION_SEEDS, TEST_SEEDS, PUBLIC_SEEDS
from .environment import build_vec_env


def evaluate_tracks(policy, config: PPOConfig, seeds, *, split="validation", video_path=None,
                    max_episode_steps=3000):
    """Score complete episodes from terminal metadata, never from a reset image.

    A shortened debug limit is explicitly labeled and is not a competition score.
    Repeating a deterministic seed is a reproducibility check, not new coverage.
    """
    pools = {"training": TRAINING_SEEDS, "validation": VALIDATION_SEEDS,
             "test": TEST_SEEDS, "public": PUBLIC_SEEDS}
    seeds = tuple(seeds)
    if not seeds or split not in pools or not set(seeds).issubset(pools[split]):
        raise ValueError("Choose seeds belonging to the named nonempty split")
    if video_path is not None and len(seeds) != 1:
        raise ValueError("Record one track per video")
    rows = []
    for seed in seeds:
        env = build_vec_env(config, (seed,), n_envs=1, random_order=False,
                            render_mode="rgb_array" if video_path else None,
                            max_episode_steps=max_episode_steps)
        writer = None
        try:
            if video_path:
                import imageio.v2 as imageio
                from PIL import Image
                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(video_path), fps=50 / config.action_repeat)
            observation = env.reset()
            reward_sum = 0.0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, rewards, dones, infos = env.step(action)
                reward_sum += float(rewards[0])
                if writer is not None and not dones[0]:
                    writer.append_data(np.asarray(Image.fromarray(env.envs[0].render()).resize((480, 320))))
                if dones[0]:
                    info = infos[0]
                    if not info.get("TimeLimit.truncated", False) and "lap_finished" not in info:
                        raise RuntimeError("Missing lap metadata: use the pinned Gymnasium version")
                    finished = bool(info.get("lap_finished", False))
                    steps = int(info["raw_steps"])
                    rows.append({"split": split, "track_seed": int(seed), "finished": finished,
                                 "raw_steps": steps, "lap_seconds": steps / 50 if finished else None,
                                 "penalized_seconds": steps / 50 if finished else 70.0,
                                 "reward": reward_sum, "raw_step_limit": max_episode_steps})
                    break
        finally:
            if writer is not None:
                writer.close()
            env.close()
    return rows


def summarize(rows):
    """Count finishes and include DNF penalties; finisher-only means hide failures."""
    if not rows:
        raise ValueError("Cannot summarize zero episodes")
    finishes = [r["lap_seconds"] for r in rows if r["finished"]]
    return {"tracks": len(rows), "completion_rate": len(finishes) / len(rows),
            "mean_penalized_seconds": float(np.mean([r["penalized_seconds"] for r in rows])),
            "mean_finishing_seconds": float(np.mean(finishes)) if finishes else None,
            "mean_reward": float(np.mean([r["reward"] for r in rows]))}


def selection_key(summary, rows):
    """Primary race score; time-adjusted reward breaks ties without early-crash bias."""
    progress = np.mean([r["reward"] + 0.1 * r["raw_steps"] for r in rows])
    return summary["mean_penalized_seconds"], -float(progress)


def checkpoint_fingerprint(path):
    """Hash the policy tensors' file bytes to distinguish weights at the same counter."""
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if "policy.pth" in archive.namelist():
            return hashlib.sha256(archive.read("policy.pth")).hexdigest()
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation", "test", "public"), default="validation")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Choose one explicit seed from the split")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-raw-steps", type=int, default=3000, help="Shorter is debug-only")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    pools = {"training": TRAINING_SEEDS, "validation": VALIDATION_SEEDS, "test": TEST_SEEDS, "public": PUBLIC_SEEDS}
    if (args.seed is None and not 1 <= args.count <= len(pools[args.split])) or args.max_raw_steps < 1:
        parser.error("count and raw step limit must be valid positive values")
    seeds = (args.seed,) if args.seed is not None else pools[args.split][:args.count]
    if not set(seeds).issubset(pools[args.split]):
        parser.error("explicit seed must belong to the named split")
    if args.video and (len(seeds) != 1 or len(args.checkpoints) != 1):
        parser.error("video requires --seed and one checkpoint")
    if args.output.resolve() in {p.resolve() for p in args.checkpoints}:
        parser.error("output must not overwrite a checkpoint")
    if args.video and (args.video.resolve() == args.output.resolve() or args.video.resolve() in {p.resolve() for p in args.checkpoints}):
        parser.error("video must not overwrite reports or source checkpoints")
    reports = []
    for path in args.checkpoints:
        policy, config, transitions = load_policy(path, args.device)
        rows = []
        for seed in seeds:
            row = evaluate_tracks(policy, config, (seed,), split=args.split,
                                  video_path=args.video, max_episode_steps=args.max_raw_steps)[0]
            rows.append(row)
            print(json.dumps({"checkpoint": str(path), **row}), flush=True)
        report = {"checkpoint": str(path), "timesteps": transitions,
                  "policy_sha256": checkpoint_fingerprint(path), "config": config.to_dict(),
                  "summary": summarize(rows), "tracks": rows,
                  "device": str(next(policy.parameters()).device),
                  "runtime_versions": {"python": sys.version, **{name: importlib.metadata.version(name)
                      for name in ("gymnasium", "stable-baselines3", "torch", "numpy", "pillow")}}}
        reports.append(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2))
        print("Summary:", report["summary"], flush=True)
    if len(reports) > 1:
        winner = min(reports, key=lambda item: selection_key(item["summary"], item["tracks"]))
        print("Selected from this comparison:", winner["checkpoint"])


if __name__ == "__main__":
    main()
