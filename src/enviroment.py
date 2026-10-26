import gymnasium as gym


def make_env(render_mode=None):
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        max_episode_steps=3000,
        render_mode=render_mode,
    )
    return env

env = make_env(render_mode="human") #to visualize the environment, set render_mode to "human"

observation, info = env.reset(seed=202)

print("Image shape:", observation.shape)
print("Pixel type:", observation.dtype)
print("Action minimums:", env.action_space.low)
print("Action maximums:", env.action_space.high)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()