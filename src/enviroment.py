import gymnasium as gym


def make_env(render_mode=None):
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        max_episode_steps=3000,
        render_mode=render_mode,
    )
    return env