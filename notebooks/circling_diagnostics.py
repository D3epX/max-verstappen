"""Diagnose an existing CarRacing PPO checkpoint without updating its weights.

This file is also embedded in both training notebooks. It needs their existing
build_vec_env function, supplied as an argument. It does not require src/ imports.
"""

import json
from pathlib import Path

import numpy as np
import torch


def diagnose_circling(policy, build_env, training_seeds, validation_seeds, output_dir):
    """Compare deterministic and sampled actions on three tracks per split.

    Returns a JSON-compatible report and saves it without overwriting previous
    reports. Uses training-pool tracks, which may not all have been encountered
    during a short training run. Sampled evaluations are diagnostic, not the
    primary checkpoint-selection protocol. Each mode is tested once per track.
    """
    if policy.use_sde or policy.policy.squash_output:
        raise ValueError("This diagnostic expects our ordinary unsquashed Gaussian PPO policy")
    training_seeds = tuple(training_seeds)[:3]
    validation_seeds = tuple(validation_seeds)[:3]
    if not training_seeds or not validation_seeds:
        raise ValueError("Supply nonempty training and validation seed groups")
    if set(training_seeds) & set(validation_seeds):
        raise ValueError("Training and validation groups must be separate")

    rows = []
    device = policy.device
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    previous_training_mode = policy.policy.training
    gaussian_std = policy.policy.log_std.detach().exp().cpu().numpy().tolist()
    print("Checkpoint transitions:", policy.num_timesteps, flush=True)
    print("Gaussian standard deviations [steering, gas, brake]:", gaussian_std, flush=True)

    # Restore the Torch RNG after diagnostic sampling so it does not advance
    # the training policy's exploration RNG. Environments are separate too.
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for split, seeds in (("training_pool", training_seeds), ("validation", validation_seeds)):
                for deterministic in (True, False):
                    for index, seed in enumerate(seeds):
                        if device.type == "cuda":
                            with torch.cuda.device(device):
                                torch.cuda.manual_seed(12345 + index)
                        else:
                            torch.manual_seed(12345 + index)
                        env = build_env((int(seed),), n_envs=1, random_order=False)
                        actions = []
                        probabilities_of_clipping = []
                        examples = []
                        rewards = []
                        try:
                            observation = env.reset()
                            assert observation.shape == (1, 4, 84, 84)
                            assert observation.dtype == np.uint8
                            while True:
                                decision = len(actions)
                                if decision % 40 == 0:
                                    with torch.no_grad():
                                        tensor, _ = policy.policy.obs_to_tensor(observation)
                                        gaussian = policy.policy.get_distribution(tensor).distribution
                                        low = torch.as_tensor(policy.action_space.low, device=device)
                                        high = torch.as_tensor(policy.action_space.high, device=device)
                                        # Exact conditional probability that a Gaussian
                                        # sample lies outside each legal control range.
                                        clipping = gaussian.cdf(low) + 1 - gaussian.cdf(high)
                                        probabilities_of_clipping.append(clipping[0].cpu().numpy())
                                        if len(examples) < 6:
                                            examples.append({
                                                "decision": decision,
                                                "unclipped_mean": gaussian.mean[0].cpu().numpy().tolist(),
                                                "std": gaussian.stddev[0].cpu().numpy().tolist(),
                                            })
                                action, _ = policy.predict(observation, deterministic=deterministic)
                                if not np.isfinite(action).all():
                                    raise ValueError("Non-finite policy action")
                                assert np.all(action >= policy.action_space.low)
                                assert np.all(action <= policy.action_space.high)
                                actions.append(action[0].copy())
                                observation, reward, done, infos = env.step(action)
                                rewards.append(float(reward[0]))
                                if done[0]:
                                    info = infos[0]
                                    if not info.get("TimeLimit.truncated", False) and "lap_finished" not in info:
                                        raise RuntimeError("Missing lap completion metadata")
                                    completed = bool(info.get("lap_finished", False))
                                    raw_steps = int(info["raw_steps"])
                                    controls = np.asarray(actions)
                                    rows.append({
                                        "split": split, "seed": int(seed),
                                        "deterministic": deterministic,
                                        "finished": completed, "raw_steps": raw_steps,
                                        "penalized_seconds": raw_steps / 50 if completed else 70.0,
                                        "reward": float(sum(rewards)),
                                        "executed_action_mean": controls.mean(axis=0).tolist(),
                                        "fraction_abs_steering_over_0_8": float(np.mean(np.abs(controls[:, 0]) > 0.8)),
                                        "fraction_both_pedals_over_0_1": float(np.mean((controls[:, 1] > 0.1) & (controls[:, 2] > 0.1))),
                                        "mean_probability_of_sample_clipping": np.mean(probabilities_of_clipping, axis=0).tolist(),
                                        "first_mean_examples": examples,
                                    })
                                    print(
                                        split, "deterministic" if deterministic else "sampled", seed,
                                        "finished:", completed, "reward:", round(sum(rewards), 2),
                                        flush=True,
                                    )
                                    break
                        finally:
                            env.close()
    finally:
        policy.policy.set_training_mode(previous_training_mode)

    summaries = []
    for split in ("training_pool", "validation"):
        for deterministic in (True, False):
            subset = [row for row in rows if row["split"] == split and row["deterministic"] == deterministic]
            summaries.append({
                "split": split, "deterministic": deterministic, "episodes": len(subset),
                "completion_rate": float(np.mean([row["finished"] for row in subset])),
                "mean_reward": float(np.mean([row["reward"] for row in subset])),
                "mean_penalized_seconds": float(np.mean([row["penalized_seconds"] for row in subset])),
            })
    report = {
        "checkpoint_transitions": int(policy.num_timesteps),
        "gaussian_std": gaussian_std,
        "summary": summaries,
        "episodes": rows,
        "limitations": "Three tracks per group, one rollout per action mode. Training-pool membership does not prove prior exposure. Differences are diagnostic, not causal proof.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A fresh filename preserves earlier diagnostics instead of overwriting them.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"circling_diagnostics_{policy.num_timesteps}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summaries, indent=2))
    print("Report saved:", path)
    return report
