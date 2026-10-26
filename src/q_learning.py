"""A tabular Q-learning example using Gymnasium's Blackjack environment.

This small program is for learning Q-learning. It is not the CarRacing agent:
Blackjack has a small, discrete observation and action space, while CarRacing
provides images and continuous steering and pedal controls.
"""

from collections import defaultdict, deque
from time import monotonic

import gymnasium as gym
import numpy as np
from tqdm import tqdm


Observation = tuple[int, int, bool]


class BlackjackAgent:
    """Learn action values for Blackjack with tabular Q-learning."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ) -> None:
        self.env = env
        self.q_values: defaultdict[Observation, np.ndarray] = defaultdict(
            lambda: np.zeros(env.action_space.n, dtype=np.float64)
        )

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_errors: list[float] = []

    def get_action(self, observation: Observation, *, explore: bool = True) -> int:
        """Choose a random action sometimes, otherwise choose the best known one."""
        if explore and np.random.random() < self.epsilon:
            return int(self.env.action_space.sample())

        return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        terminated: bool,
        next_observation: Observation,
    ) -> None:
        """Update one Q-value using the reward and the next observation."""
        if terminated:
            best_future_value = 0.0
        else:
            best_future_value = float(np.max(self.q_values[next_observation]))

        target = reward + self.discount_factor * best_future_value
        error = target - self.q_values[observation][action]
        self.q_values[observation][action] += self.learning_rate * error
        self.training_errors.append(float(error))

    def decay_epsilon(self) -> None:
        """Reduce random exploration, while keeping it above its minimum."""
        self.epsilon = max(
            self.final_epsilon,
            self.epsilon - self.epsilon_decay,
        )


def train_agent(
    agent: BlackjackAgent,
    env: gym.Env,
    duration_seconds: float,
) -> None:
    """Train for roughly the requested time, finishing each hand in progress."""
    deadline = monotonic() + duration_seconds
    recent_rewards: deque[float] = deque(maxlen=100)
    with tqdm(desc="Training Blackjack agent", unit="hand") as progress:
        while monotonic() < deadline:
            observation, _ = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action = agent.get_action(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)

                agent.update(
                    observation,
                    action,
                    float(reward),
                    terminated,
                    next_observation,
                )
                observation = next_observation
                episode_reward += float(reward)

            agent.decay_epsilon()
            recent_rewards.append(episode_reward)
            progress.update(1)

            # Refresh the learning summary every 100 hands to keep the display readable.
            if progress.n % 100 == 0:
                recent = np.asarray(recent_rewards)
                progress.set_postfix(
                    epsilon=f"{agent.epsilon:.2f}",
                    win_rate=f"{np.mean(recent > 0):.0%}",
                    q_states=len(agent.q_values),
                )


def evaluate_agent(
    agent: BlackjackAgent,
    env: gym.Env,
    num_episodes: int = 1_000,
) -> None:
    """Measure the learned policy without random exploration or learning."""
    rewards = []

    for _ in tqdm(range(num_episodes), desc="Evaluating agent"):
        observation, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.get_action(observation, explore=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)

        rewards.append(episode_reward)

    rewards_array = np.asarray(rewards)
    win_rate = np.mean(rewards_array > 0)
    print(f"\nEvaluation over {num_episodes:,} hands")
    print(f"Win rate (excluding draws): {win_rate:.1%}")
    print(f"Average reward: {np.mean(rewards_array):.3f}")
    print(f"Reward standard deviation: {np.std(rewards_array):.3f}")


def main() -> None:
    training_seconds = 30
    env = gym.make("Blackjack-v1", sab=False)

    agent = BlackjackAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=1.0 / (training_seconds * 50),
        final_epsilon=0.1,
        discount_factor=0.95,
    )

    try:
        train_agent(agent, env, duration_seconds=training_seconds)
        evaluate_agent(agent, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
