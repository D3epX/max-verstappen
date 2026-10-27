"""Numerical regression checks for PPO math, episode boundaries and portability."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from src.ppo.config import PPOConfig
from src.ppo.gae import compute_gae
from src.ppo.loss import ppo_loss
from src.ppo.model import make_model
from src.ppo.checkpoint import save_checkpoint, resume_model, load_policy
from src.ppo.evaluate import selection_key


class TinyImages(gym.Env):
    """Short truncating episodes exercise bootstrap and automatic reset boundaries."""
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        self.action_space = gym.spaces.Box(np.array([-1, 0, 0], np.float32), np.ones(3, np.float32))
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros((4, 84, 84), np.uint8), {}

    def step(self, action):
        self.steps += 1
        obs = np.full((4, 84, 84), 40 * self.steps, np.uint8)
        return obs, float(action[1] - action[2]), False, self.steps == 3, {}


class PPOChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gae_matches_sb3_and_stops_at_reset(self):
        config = PPOConfig(n_steps=8, batch_size=8)
        buffer = RolloutBuffer(8, gym.spaces.Box(0, 255, (4, 84, 84), np.uint8),
                               gym.spaces.Box(-1, 1, (3,)), n_envs=2,
                               gamma=config.gamma, gae_lambda=config.gae_lambda)
        rng = np.random.default_rng(3)
        buffer.rewards = rng.normal(size=(8, 2)).astype(np.float32)
        buffer.values = rng.normal(size=(8, 2)).astype(np.float32)
        buffer.episode_starts = np.zeros((8, 2), np.float32)
        buffer.episode_starts[4, 0] = 1
        last = torch.tensor([1., 2.]);dones = np.array([False, True])
        expected = compute_gae(buffer.rewards, buffer.values, buffer.episode_starts,
                               last.numpy(), dones, config.gamma, config.gae_lambda)
        buffer.compute_returns_and_advantage(last, dones)
        np.testing.assert_allclose(expected[0], buffer.advantages)
        np.testing.assert_allclose(expected[1], buffer.returns)
        # A reset cuts off a large next-episode reward entirely.
        advantages, _ = compute_gae([[1], [100]], [[0], [0]], [[0], [1]], [0], [True])
        self.assertEqual(advantages[0, 0], 1.)
        # A timeout's already-bootstrapped reward survives that same boundary.
        advantages, _ = compute_gae([[1 + .99 * 5]], [[2]], [[0]], [999], [True])
        self.assertAlmostEqual(float(advantages[0, 0]), 3.95, places=5)

    def test_clipping_objective_and_target_detachment(self):
        old = torch.zeros(2, requires_grad=True)
        target = torch.tensor([2., 4.], requires_grad=True)
        values = torch.zeros(2, requires_grad=True)
        new = torch.tensor([1.5, .5]).log().requires_grad_()
        loss = ppo_loss(values, new, torch.ones(2), old, torch.tensor([1., -1.]), target,
                        normalize_advantage=False)
        self.assertAlmostEqual(loss.actor.item(), -.2, places=6)
        self.assertAlmostEqual(loss.critic.item(), 10.)
        self.assertAlmostEqual(loss.total.item(), 4.8, places=6)
        loss.total.backward()
        self.assertIsNone(old.grad);self.assertIsNone(target.grad)
        self.assertGreater(values.grad.abs().sum().item(), 0)

    def test_update_parity_and_checkpoint_resume(self):
        config = PPOConfig(n_steps=8, batch_size=8, n_epochs=2)
        env = DummyVecEnv([TinyImages, TinyImages])
        reference_env = DummyVecEnv([TinyImages, TinyImages])
        try:
            model = make_model(config, env, "cpu")
            model.set_logger(configure(None, []))
            model.learn(16, progress_bar=False)
            self.assertEqual(model.num_timesteps, 16)
            self.assertTrue(np.isfinite(model.rollout_buffer.returns).all())
            reference = make_model(config, reference_env, "cpu")
            reference.set_logger(configure(None, []))
            reference.policy.load_state_dict(model.policy.state_dict())
            reference.policy.optimizer.load_state_dict(copy.deepcopy(model.policy.optimizer.state_dict()))
            reference.rollout_buffer = copy.deepcopy(model.rollout_buffer)
            reference._current_progress_remaining = model._current_progress_remaining
            reference._n_updates = model._n_updates
            # Force real optimizer steps in the parity check, not identical early exits.
            model.target_kl = reference.target_kl = None
            before = {k: v.clone() for k, v in model.policy.state_dict().items()}
            state = np.random.get_state()
            model.train()
            np.random.set_state(state)
            # Independent reference: invoke SB3's implementation, not our override.
            PPO.train(reference)
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items()))
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, reference.policy.state_dict()[key], rtol=1e-5, atol=1e-6)
            with tempfile.TemporaryDirectory() as tmp:
                path = save_checkpoint(model, Path(tmp) / "model.zip")
                policy, loaded_config, steps = load_policy(path)
                self.assertEqual(loaded_config, config);self.assertEqual(steps, 16)
                obs = np.zeros((1, 4, 84, 84), np.uint8)
                np.testing.assert_allclose(model.predict(obs, deterministic=True)[0],
                                           policy.predict(obs, deterministic=True)[0])
                resumed = resume_model(path, reference_env, "cpu")
                self.assertEqual(resumed.num_timesteps, 16)
                self.assertEqual(resumed._n_updates, model._n_updates)
                self.assertEqual(len(resumed.policy.optimizer.state), len(model.policy.optimizer.state))
                resumed.set_logger(configure(None, []))
                resumed.learn(16, reset_num_timesteps=False, progress_bar=False)
                self.assertEqual(resumed.num_timesteps, 32)
        finally:
            env.close();reference_env.close()

    def test_collector_matches_sb3_with_timeout_bootstrap(self):
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
        config = PPOConfig(n_steps=8, batch_size=8)
        env = DummyVecEnv([TinyImages, TinyImages])
        reference_env = DummyVecEnv([TinyImages, TinyImages])
        try:
            model = make_model(config, env, "cpu")
            reference = make_model(config, reference_env, "cpu")
            reference.policy.load_state_dict(model.policy.state_dict())
            model.set_logger(configure(None, []))
            reference.set_logger(configure(None, []))
            _, callback = model._setup_learn(16)
            _, reference_callback = reference._setup_learn(16)
            reference.rollout_buffer = RolloutBuffer(
                8, reference.observation_space, reference.action_space,
                device="cpu", n_envs=2, gamma=config.gamma, gae_lambda=config.gae_lambda)
            rng_state = torch.get_rng_state()
            model.collect_rollouts(env, callback, model.rollout_buffer, 8)
            torch.set_rng_state(rng_state)
            OnPolicyAlgorithm.collect_rollouts(reference, reference_env, reference_callback,
                                              reference.rollout_buffer, 8)
            for name in ("observations", "actions", "values", "log_probs", "rewards", "advantages", "returns"):
                np.testing.assert_allclose(getattr(model.rollout_buffer, name),
                                           getattr(reference.rollout_buffer, name), rtol=1e-5, atol=1e-6)
        finally:
            env.close();reference_env.close()

    def test_dnf_selection_does_not_reward_early_crash(self):
        summary = {"mean_penalized_seconds": 70.}
        timeout = selection_key(summary, [{"reward": -200, "raw_steps": 3000}])
        crash = selection_key(summary, [{"reward": -100, "raw_steps": 100}])
        self.assertLess(timeout, crash)
        finish = selection_key({"mean_penalized_seconds": 60}, [{"reward": 1, "raw_steps": 3000}])
        self.assertLess(finish, timeout)


if __name__ == "__main__":
    unittest.main()
