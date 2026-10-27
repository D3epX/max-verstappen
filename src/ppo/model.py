"""Use SB3's tested policy/distribution infrastructure with explicit local PPO steps."""
from stable_baselines3 import PPO
from .config import PPOConfig
from .policy import policy_kwargs
from .rollout import GAERolloutBuffer, collect_rollout
from .update import update_policy


class LocalPPO(PPO):
    """Own collection, GAE, loss and update; SB3 supplies Adam/policy/save/learn loop.

    This is continuous Box-action PPO for this project, not a generic reimplementation
    of every SB3 algorithm or support for every observation/action space.
    """
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        return collect_rollout(self, env, callback, rollout_buffer, n_rollout_steps)

    def train(self):
        update_policy(self)


def make_model(config: PPOConfig, env, device="auto", log_dir=None):
    """Initialize a fresh compatible policy and optimizer; never loads a checkpoint."""
    model = LocalPPO(
        "CnnPolicy", env, device=device, seed=config.run_seed,
        policy_kwargs=policy_kwargs(config), rollout_buffer_class=GAERolloutBuffer,
        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
        learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm, target_kl=config.target_kl,
        use_sde=config.use_sde, sde_sample_freq=config.sde_sample_freq,
        tensorboard_log=None if log_dir is None else str(log_dir), verbose=1,
    )
    model.experiment_config = config.to_dict()
    return model
