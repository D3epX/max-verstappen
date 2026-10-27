"""Repeated minibatch optimization of one fresh PPO rollout.

SB3-compatible update semantics; see its MIT-licensed PPO implementation:
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
"""
import numpy as np
import torch
from stable_baselines3.common.utils import explained_variance
from .loss import ppo_loss


def update_policy(model):
    """Backpropagate the combined objective and stop overly large policy changes.

    Epochs reuse this rollout only. Adam updates the heads, shared CNN and learned
    exploration scales; the old rollout targets never receive gradients.
    """
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    clip = model.clip_range(model._current_progress_remaining)
    clip_vf = None if model.clip_range_vf is None else model.clip_range_vf(model._current_progress_remaining)
    history = {name: [] for name in ("entropy_loss", "policy_gradient_loss", "value_loss", "approx_kl", "clip_fraction")}
    stopped = False
    for epoch in range(model.n_epochs):
        for batch in model.rollout_buffer.get(model.batch_size):
            values, log_prob, entropy = model.policy.evaluate_actions(batch.observations, batch.actions)
            terms = ppo_loss(
                values, log_prob, entropy, batch.old_log_prob, batch.advantages, batch.returns,
                clip_range=clip, vf_coef=model.vf_coef, ent_coef=model.ent_coef,
                normalize_advantage=model.normalize_advantage,
                old_values=batch.old_values, clip_range_vf=clip_vf,
            )
            for name, value in (("entropy_loss", terms.entropy), ("policy_gradient_loss", terms.actor),
                                ("value_loss", terms.critic), ("approx_kl", terms.approx_kl),
                                ("clip_fraction", terms.clip_fraction)):
                history[name].append(value.item())
            if not torch.isfinite(terms.total):
                raise FloatingPointError("non-finite PPO loss; preserve the earlier checkpoint")
            if model.target_kl is not None and terms.approx_kl.item() > 1.5 * model.target_kl:
                stopped = True
                if model.verbose:
                    print(f"Stopping update epoch {epoch}: approximate KL {terms.approx_kl.item():.4f}")
                break
            model.policy.optimizer.zero_grad()
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()
        model._n_updates += 1
        if stopped:
            break
    metrics = {name: float(np.mean(values)) for name, values in history.items()}
    metrics.update(
        loss=terms.total.item(),
        explained_variance=float(explained_variance(model.rollout_buffer.values.flatten(), model.rollout_buffer.returns.flatten())),
        std=model.policy.log_std.exp().mean().item(), n_updates=model._n_updates,
        clip_range=clip, kl_early_stop=float(stopped),
    )
    for name, value in metrics.items():
        model.logger.record(f"train/{name}", value)
    return metrics
