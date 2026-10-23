# CarRacing AI Challenge

> Train your own AI driver. Make it fast. Understand every part of it.

|  |  | 
| ----- | ----- | 
| **Environment** | Gymnasium `CarRacing-v3` | 
| **Format** | Individual competition (collaboration allowed with limits, see section 5) | 
| **Goal** | Train an AI model that completes a lap in the **fastest time** | 
| **Rounds** | 4 rounds: 3 known tracks + 1 mystery track | 
| **Final check** | Mandatory knowledge interview; failing it means elimination | 

## 1. What is this competition?

Each participant trains their own AI model to drive a car in the Gymnasium **CarRacing** environment. The car sees the world only through a small top-down image (96×96 pixels) and has to learn to steer, accelerate and brake by itself.

The organizer runs every submitted PyTorch model on the same tracks with the official evaluation script, and the leaderboard ranks the fastest drivers.

But speed alone is not enough. This competition is also about **learning**. Every participant must understand neural networks, reinforcement learning, and the architecture they used. Anyone who cannot explain their own work is eliminated, however fast their car is.

## 2. Objectives

### Main objective

Train and optimize an AI model that completes each track in the **shortest possible time** without going off the map.

### Side objectives (mandatory for everyone)

Every participant must build real knowledge in:

1. **Neural networks**: neurons, layers, activation functions, loss, gradient descent, backpropagation, CNNs for images.

2. **Reinforcement learning**: agent, environment, state, action, reward, policy, value function, exploration vs exploitation, discount factor, and the algorithm they chose.

3. **Model architecture**: the exact network they used, why it has that shape, what the input and output are, and why they made each design choice.

Participants may go as deep as they want on the technical side. A simple model that you fully understand beats a complex one you copied.

## 3. Daily Meets & Development Journal

* **Daily Meets**: Regular check-ins where participants discuss progress, blockers, and share insights.

* **Mandatory GitHub Journal**: Every member must create a dedicated GitHub repository for their work. Inside the repository, you **MUST** maintain a `.md` file named **`journal.md`**. This document is a critical part of your evaluation and must log your complete development journey, including:
  * Your overall approach and strategy.
  * Problems, bugs, and technical hurdles faced.
  * How you debugged and solved issues.
  * Architectural ideas, experiments, and thoughts.
  * Key things learned about RL and Neural Networks throughout the project.

## 4. The tracks

CarRacing generates a track from a **seed** number, so the same seed always gives exactly the same track. The 3 known tracks were chosen to get progressively harder:

| Round | Track seed | Difficulty | Track tiles | Character | 
| ----- | ----- | ----- | ----- | ----- | 
| **Round 1** | `202` | 🟢 Easy | 276 | Smooth and flowing, long straights, gentle corners | 
| **Round 2** | `42` | 🟡 Medium | 283 | Mixed: fast straights plus a tight hairpin section | 
| **Round 3** | `203` | 🔴 Hard | 360 | Longest track, star-shaped, many sharp corners in a row | 
| **Round 4** | ❓ Mystery | ❓ | ❓ | Chosen by the organizer **after** submissions close | 

To see a track:

```python
import gymnasium as gym
env = gym.make("CarRacing-v3", render_mode="human")
env.reset(seed=202)   # change to 42 or 203
```

### About the mystery round

* The organizer picks the mystery seed **only after the submission deadline**, so nobody can train on it.

* It will not be 202, 42 or 203.

* It exists to reward agents that **really learned to drive**, not agents that memorized 3 tracks. Train on many random seeds, not only the 3 known ones.

## 5. Collaboration policy

This is an **individual** competition: every person submits their own AI and is ranked alone. Collaboration is allowed, but only to help each other learn.

### ✅ Allowed

* Discussing concepts, algorithms, papers and tutorials together.

* Explaining things to each other (NN, RL, how PPO works, ...).

* Helping someone debug an installation or an error message.

* Sharing public learning resources.

### ❌ Not allowed

* Sharing or copying training code, model files or trained weights.

* Submitting a model trained (fully or partly) by someone else.

* Training a model "together" and submitting it twice.

* Using pre-trained CarRacing models downloaded from the internet.

## 6. Technical rules

### Environment

* Environment: `CarRacing-v3` from Gymnasium (version **1.x**), default settings, **continuous actions**.

* Install: `pip install "gymnasium[box2d]"`

* Observation: an RGB image, `numpy.uint8`, shape `(96, 96, 3)`.

* Action: `[steering, gas, brake]`, with steering in `[-1, 1]` and gas and brake in `[0, 1]`.

* If you trained with discrete actions (`continuous=False`), convert them to the continuous format during inference.

### ⚠️ Important: the time limit

By default, `gym.make("CarRacing-v3")` stops the episode after **1000 steps (20 seconds)**. That is too short to finish the hard tracks. The official evaluation allows **3000 steps (60 seconds)**, so train with the same limit:

```python
env = gym.make("CarRacing-v3", max_episode_steps=3000)
```

### What your AI may and may not do

* ✅ Any learning method: PPO, SAC, DQN, A2C, evolution strategies, imitation learning, ...

* ✅ Any library: Stable-Baselines3, PyTorch, TensorFlow, JAX, ...

* ✅ Preprocessing: grayscale, cropping, frame stacking, frame skipping, normalization.

* ✅ Custom reward shaping during **training**.

* ❌ The final driver must be a **trained neural network**. Hand-written `if/else` driving rules are not allowed as the final agent.

* ❌ The agent may only use the image it receives. Reading the game's internal state (car position, speed, track coordinates through `env.unwrapped`) is not allowed at evaluation.

* ❌ No internet access during evaluation.

## 7. Submission Requirements (VERY IMPORTANT)

Submissions that do not strictly meet these requirements will be rejected. Every participant must submit:

1. 🚨 **PyTorch Model File (`.pt` or `.pth`)**: The exact trained PyTorch model weights used for evaluation.
2. 🚨 **GitHub Repository Link**: A link to your public/accessible GitHub repository containing your full source code.
3. 🚨 **`journal.md` File**: Located directly inside your GitHub repository, documenting your complete journey, approach, problems faced, solutions implemented, and key takeaways.

The organizer will run all automated tests and manage the official leaderboard using these submitted files.

## 8. Scoring

### Lap time per round

* Each agent drives **3 runs** on each track.

* The lap time of one run = race time when the lap is completed (CarRacing counts a lap as complete when the car has driven over at least 95% of the track tiles and is back at the start).

* A run that does not finish (drives off the map, or is still racing after 60 s) is a **DNF** and counts as **70 seconds** (60 s + 10 s penalty).

* **Round time = average of the 3 runs.** Lower is better.

### Points per round

Participants are ranked by round time in each round and receive points:

| Position | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Points** | 25 | 18 | 15 | 12 | 10 | 8 | 6 | 4 | 2 | 1 | 

A participant with a DNF in all 3 runs of a round gets **0 points** for that round.

* Rounds 1, 2 and 3: normal points.

* **Round 4 (mystery track): points count double (×2)**, because it shows whether your AI truly learned to drive.

### Final ranking

**Final score = R1 + R2 + R3 + (2 × R4)**

Tie-breakers, in order:

1. Better time on the mystery track.

2. Lower total time across all 4 rounds.

Only participants who **pass the knowledge interview** appear in the final ranking.

## 9. Knowledge interview (mandatory)

Every participant has a **10 to 15 minute interview** with the jury after the races. The goal is simple: **show that you know what you did, how you did it, and why.**

### What the jury will ask about

**A. Neural networks**

* What is a neuron? What does an activation function do? Why do we need non-linearity?

* What is a loss function? How does gradient descent work? What is backpropagation?

* Why use a CNN for images instead of a fully connected network?

* What are overfitting and the learning rate?

**B. Reinforcement learning**

* Explain agent, environment, state, action, reward and episode using CarRacing.

* What is a policy? What is a value function?

* What is the discount factor (gamma), and what happens if it is 0 or 0.999?

* What is exploration vs exploitation, and how does your algorithm explore?

* How does your algorithm learn? (For example: what does "clipping" mean in PPO? What is a replay buffer in DQN/SAC?)

**C. Your own work & journal**

* Walk us through your network, layer by layer. What goes in and what comes out?

* Explain the development journey and key entries documented in your `journal.md`.

* What obstacles/problems did you encounter, and how did you overcome them?

* Why did you choose your specific approach and architecture?

* What preprocessing did you do, and why?

* Did you change the reward? What problem did that solve?

* What failed during training, and how did you fix it?

* How did you prepare for the mystery track?

* If you had one more day, what would you improve?

### How it is judged

Each area is scored from 0 to 3:

| Score | Meaning | 
| ----- | ----- | 
| **0** | Cannot answer, or clearly does not understand | 
| **1** | Basic, incomplete understanding | 
| **2** | Good understanding, explains clearly | 
| **3** | Deep understanding, can reason beyond the basics | 

### ❌ Elimination

A participant is **judged unworthy and eliminated** if **any** of the following is true:

* Scores **0 in any area** (A, B or C).

* Total interview score is **below 4 out of 9**.

* **Cannot explain their own submitted code, journal, or PyTorch model** (area C is the most important).

* Evidence of breaking the collaboration rules (section 5).

The jury's decision is final. Eliminated participants are removed from the final ranking, whatever their lap times.

## 10. Recommended learning resources

* **Gymnasium CarRacing documentation**: https://gymnasium.farama.org/environments/box2d/car_racing/

* **Hugging Face Deep RL Course** (free, beginner friendly): https://huggingface.co/learn/deep-rl-course

* **OpenAI Spinning Up** (clear explanations of RL algorithms): https://spinningup.openai.com

* **Stable-Baselines3** (ready-to-use PPO, SAC, DQN): https://stable-baselines3.readthedocs.io

* **3Blue1Brown, "Neural Networks" series on YouTube**: the best visual introduction to neural networks.

* **Sutton & Barto, *Reinforcement Learning: An Introduction*** (free online): http://incompleteideas.net/book/the-book-2nd.html

### Tips for beginners

* Get **something** driving early, then improve it. Don't wait for the perfect plan.

* Common winning recipe: grayscale + frame stacking (4 frames) + a small CNN + PPO.

* Train on **random seeds**, and only use 202, 42 and 203 to test.

* Save checkpoints often, and keep the best one, not the last one.

* Keep your GitHub `journal.md` updated daily as you experiment. It will serve as your primary preparation for the oral knowledge interview.

## Appendix A: Official evaluation script reference

This is the general script structure used to evaluate submitted PyTorch models:

```python
"""Official evaluation script reference - CarRacing AI Challenge."""
import gymnasium as gym
import numpy as np
import torch

def evaluate_model(model_path, seed, max_steps=3000):
    env = gym.make("CarRacing-v3", max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    
    # Load PyTorch model file (.pt / .pth)
    model = torch.load(model_path)
    model.eval()
    
    done = False
    info = {}
    while not done:
        # Process observation and run model inference
        # action = ...
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    lap_time = env.unwrapped.t if info.get("lap_finished", False) else None
    env.close()
    return lap_time
```

**Good luck, and may the fastest (and smartest) driver win! 🏁**