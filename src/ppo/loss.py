"""PPO's clipped actor loss, critic error, entropy term and update diagnostics."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class LossTerms:
    """Keep the scalar losses separate so logs explain the combined objective."""
    total: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(values, log_prob, entropy, old_log_prob, advantages, returns, *,
             clip_range=0.2, vf_coef=0.5, ent_coef=0.0, normalize_advantage=True,
             old_values=None, clip_range_vf=None):
    """Evaluate the SAME recorded actions under the current and old policies.

    log_prob is joint log density for continuous controls, not an action
    percentage. Old likelihoods, advantages and targets are detached. Both heads
    depend on the shared CNN, so total.backward() updates that CNN from both jobs.
    """
    values = values.flatten()
    log_prob = log_prob.flatten()
    old_log_prob = old_log_prob.detach().flatten()
    advantages = advantages.detach().flatten()
    returns = returns.detach().flatten()
    if not (values.shape == log_prob.shape == old_log_prob.shape == advantages.shape == returns.shape):
        raise ValueError("all minibatch vectors must have the same shape")
    if normalize_advantage and advantages.numel() > 1:
        # Match SB3's sample standard deviation (not population std).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages
    actor = -torch.minimum(unclipped, clipped).mean()
    predicted_values = values
    if clip_range_vf is not None:
        if old_values is None:
            raise ValueError("value clipping requires old_values")
        old_values = old_values.detach().flatten()
        predicted_values = old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf)
    critic = F.mse_loss(predicted_values, returns)
    entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
    total = actor + vf_coef * critic + ent_coef * entropy_loss
    with torch.no_grad():
        approx_kl = (ratio - 1 - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_range).float().mean()
    return LossTerms(total, actor, critic, entropy_loss, approx_kl, clip_fraction)
