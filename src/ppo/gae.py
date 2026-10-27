"""Generalized Advantage Estimation, with an explicit episode-boundary mask."""
import numpy as np


def compute_gae(rewards, values, episode_starts, last_values, dones,
                gamma=0.99, gae_lambda=0.95):
    """Return advantages and critic targets, shaped [rollout_steps, environments].

    The collector already adds gamma * V(terminal_observation) to TIMEOUT
    rewards. We must still stop the recursion at that boundary: the reset
    observation belongs to a different race. True termination gets no bootstrap.
    An unfinished rollout bootstraps from last_values instead.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    starts = np.asarray(episode_starts, dtype=np.float32)
    if rewards.ndim != 2 or rewards.shape != values.shape or starts.shape != values.shape:
        raise ValueError("rewards, values and episode_starts must share [steps, envs] shape")
    if rewards.shape[0] == 0:
        raise ValueError("rollout must contain at least one step")
    last_values = np.asarray(last_values, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    if last_values.shape != (rewards.shape[1],) or dones.shape != last_values.shape:
        raise ValueError("last_values and dones must have one entry per environment")
    advantages = np.zeros_like(rewards)
    running_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(len(rewards))):
        next_value = last_values if step == len(rewards) - 1 else values[step + 1]
        boundary = dones if step == len(rewards) - 1 else starts[step + 1]
        continues = 1.0 - boundary
        delta = rewards[step] + gamma * next_value * continues - values[step]
        running_advantage = delta + gamma * gae_lambda * continues * running_advantage
        advantages[step] = running_advantage
    # Return targets are fixed data for this update, not gradients into old values.
    return advantages, advantages + values
