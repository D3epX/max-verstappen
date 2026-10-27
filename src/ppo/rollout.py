"""Collect fresh on-policy experience and connect our GAE to SB3 storage.

The control flow follows SB3's OnPolicyAlgorithm collector (MIT license):
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py
SB3 provides the policy distributions, tensor conversion and minibatch storage;
this module makes action clipping and timeout bootstrapping explicit locally.
"""
import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from .gae import compute_gae


class GAERolloutBuffer(RolloutBuffer):
    """SB3 observation/action storage, using this project's GAE function."""
    def compute_returns_and_advantage(self, last_values, dones):
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.episode_starts,
            last_values.detach().cpu().numpy(), dones, self.gamma, self.gae_lambda,
        )


def collect_rollout(model, env, callback, buffer, n_rollout_steps):
    """Collect [steps, envs] samples, preserving unclipped actions for likelihoods.

    The game receives bounded controls. PPO stores the original Gaussian sample
    and its log density, not a clipped sample paired with the wrong likelihood.
    No gradients are built while interacting with the simulator.
    """
    if model._last_obs is None:
        raise RuntimeError("learn() must initialize observations before collection")
    model.policy.set_training_mode(False)
    buffer.reset()
    if model.use_sde:
        model.policy.reset_noise(env.num_envs)
    callback.on_rollout_start()
    for n_steps in range(n_rollout_steps):
        if model.use_sde and model.sde_sample_freq > 0 and n_steps % model.sde_sample_freq == 0:
            model.policy.reset_noise(env.num_envs)
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_as_tensor(model._last_obs, model.device))
        actions = actions.cpu().numpy()
        if model.policy.squash_output:
            executed_actions = model.policy.unscale_action(actions)
        else:
            executed_actions = np.clip(actions, model.action_space.low, model.action_space.high)
        new_obs, rewards, dones, infos = env.step(executed_actions)
        model.num_timesteps += env.num_envs
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        model._update_info_buffer(infos, dones)
        for index, done in enumerate(dones):
            info = infos[index]
            if done and info.get("TimeLimit.truncated", False) and info.get("terminal_observation") is not None:
                # VecEnv has already reset new_obs. Use the FINAL race image here.
                terminal_obs = model.policy.obs_to_tensor(info["terminal_observation"])[0]
                with torch.no_grad():
                    terminal_value = model.policy.predict_values(terminal_obs).item()
                rewards[index] += model.gamma * terminal_value
        buffer.add(model._last_obs, actions, rewards, model._last_episode_starts, values, log_probs)
        model._last_obs = new_obs
        model._last_episode_starts = dones
    with torch.no_grad():
        last_values = model.policy.predict_values(obs_as_tensor(new_obs, model.device))
    buffer.compute_returns_and_advantage(last_values, dones)
    callback.update_locals(locals())
    callback.on_rollout_end()
    return True
