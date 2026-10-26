# Development journal

## 2026-10-23 — planning and repository review

This entry records planning only. No driver has been trained and no race results have been measured.

- Initial repository: challenge guidelines, one-line README, empty requirements file, and empty `src/`.
- Compute plan: Google Colab; mkash local GPU available.
- Proposed baseline: continuous-action PPO with four grayscale frames and a compact CNN, trained from scratch on varied track seeds.
- Proposed optional comparison: a small IMPALA-style residual CNN( I will put the explanation in bottom of this journal) with the same PPO/evaluation setup.
- Validation: dependency resolution dry run succeeded for Python 3.12; Markdown local links/fences and Git whitespace checks passed. Full installation, Colab execution and model training have not been performed.
- Open questions for organizer: exact evaluator/Gymnasium version, accepted model loader/interface, repeated-run protocol, and AI coding-assistance policy.

## Write an entry after each real work session

Copy the following headings and fill them with your own observations. Do not mark planned work as completed.

### Date and goal

### What I learned and can explain

### What I implemented myself / with assistance

### Command, configuration, package versions and checkpoint

### Training transitions, raw-step convention and wall-clock compute

### Evaluation seeds, completion rate and DNF-penalized lap times

### Problems, evidence and debugging decisions

### What failed or remains uncertain

### Next experiment and why

## What I think I understand so far about the model

**Why a CNN?** The car gets a picture, not numbers telling it where the road goes. A CNN looks at small parts of the picture with the same learned filters in different places. That makes sense for road edges and turns, which can appear anywhere in the image. It also uses fewer weights than giving every pixel its own separate connection. I still need to see what the trained network actually learns; a CNN does not automatically understand the road.

**Why several frames?** One picture shows where things are, but it is hard to tell from it whether the car is moving fast or sliding. Four recent frames may give the network clues about motion. Four is a starting choice, not a proven best number for this race.

**Why PPO?** PPO learns from driving attempts. It compares what happened with what its value network expected, then updates the action policy. Its clipping rule limits how much the policy is encouraged to change in one update. That sounds useful while I am getting my first driver working, and Stable-Baselines3 provides an implementation I can study. It can still need a lot of driving experience, and clipping does not guarantee good or safe driving. [PPO paper](https://arxiv.org/abs/1707.06347).

**Why might I try a residual CNN?** A residual block adds its input to the result of a few layers. I understand this as letting the network keep some information while learning a change to it. That direct path can make a deeper network easier to train. An IMPALA-style CNN is a way to arrange these blocks for images. I would only borrow the image network; I would still use PPO to learn the driving policy. The IMPALA work and a later PPO comparison make this a reasonable experiment, but neither proves it will be faster on my tracks. [IMPALA paper](https://arxiv.org/abs/1802.01561), [DreamerV3 comparison](https://www.nature.com/articles/s41586-025-08744-2).

**How I would decide:** First get the simple CNN and PPO working. Then, if I have time, change only the CNN and give both versions similar training time. I need to compare finishes and lap times on tracks I did not train on. Until I run that test, the residual version is just an idea.

**Question I still have:** Would SAC work better than PPO here? SAC can reuse old driving experience, which might help when collecting new experience is slow. But I have not measured whether its image replay storage and training updates would fit my Colab time better. I should test it only after I have a baseline and understand how I would compare them fairly. [SAC paper](https://arxiv.org/abs/1801.01290).

## What I learned about policies and values

I learned that a **policy** is the agent's way of choosing an action from what it sees. In CarRacing, it takes the current image (or recent images) and chooses steering, gas and brake. During training it may try different actions; when I evaluate it, I can ask it to choose its most likely action.

I learned that a **value function** estimates how much future reward the agent can expect from a situation if it keeps following a particular policy. I can write this as `Vπ(s)`: the value of situation `s` under policy `π`. A **Q-value**, `Qπ(s, a)`, asks a more specific question: how much future reward can I expect if I take action `a` in situation `s`, then follow policy `π`? These are estimates, not the actual future. They include future rewards with later rewards usually discounted.

In **value-based learning**, the agent learns values for actions and uses them to choose what to do. **Q-learning** is an example: it updates its Q estimates using the reward it just received and an estimate of the best value available at the next step. It can then pick the action with the highest estimated Q-value, while still trying other actions during training. Basic Q-learning is easiest with a small list of discrete actions. Our steering, gas and brake are continuous, so applying basic Q-learning directly would be awkward.

In **policy-based learning**, the agent directly learns a policy that chooses actions. PPO is an example. It also learns a value estimate to help judge whether an action turned out better or worse than expected, so the two ideas can be used together. That helped me understand why the PPO model has both an actor and a critic.

An **optimal policy** would choose actions that give the highest expected total discounted reward for the task. If I had the exact best Q-values, I could choose the action with the highest Q-value in each situation. In practice I only learn approximations from limited experience. Also, the game's reward is not exactly the competition's lap-time score, so a policy that gets high training reward is not automatically the fastest or most reliable driver. I still need to test finishes and times on unseen tracks.

## Identifying the parts of my RL problem

I learned that before choosing an algorithm, I should be clear about what is learning, what it interacts with, what it can see, and what it can do. For this project:

- **Agent:** my AI driver. Its policy takes the image it receives and chooses how to control the car. The neural network is part of the agent; the car and track are not.
- **Environment:** Gymnasium's `CarRacing-v3` game. It generates the track, simulates what happens when the car moves, and gives the agent the next image and a reward after each action.
- **State and observation:** the full game state includes things such as the car's position and speed, but my driver does not get those numbers. What it actually receives is an RGB image with shape `(96, 96, 3)`. I can use recent images together to help it infer motion. I should call this image an **observation** rather than claim the agent sees the full state.
- **Actions:** three continuous controls: `[steering, gas, brake]`. Steering can be from `-1` to `1`; gas and brake can each be from `0` to `1`. The agent has to choose values in these ranges from its observation.

So the basic interaction I need to understand is: **image → action → environment moves → new image and reward**. The competition says the final driver can use the image, but not hidden game information such as the car's exact position or the track coordinates.

##  how I understand PPO

This is my explanation from studying the algorithm. I have not trained or measured a PPO driver yet.

**What the name means.** PPO stands for *Proximal Policy Optimization*. It is a way to train a policy from the agent's own recent experience. “Proximal” refers to keeping each policy update from changing action probabilities too aggressively. After learning from one batch of driving, the agent collects a fresh batch with its updated policy.

**What goes into the networks.** The environment gives the car a `96×96×3` image. My proposed preprocessing turns recent images into four smaller grayscale frames. A CNN turns those frames into a shorter set of useful numbers, often called features. The actor and critic use those features. They may share the CNN, but they have different final outputs and different jobs. The network weights start untrained; useful features have to be learned from experience.

**Actor, or policy network.** The actor chooses `[steering, gas, brake]`. For continuous actions, the PPO setup I am considering gives a mean for each of the three controls and learns how spread out its action choices are. During training it samples actions, so it sometimes tries different steering or pedal values. During evaluation I would usually use the mean action and apply the same action bounds. The actor does not know the best action in advance; it gradually changes which actions are likely.

**Critic, or value network.** The critic outputs one number, `V(o)`, for an observation `o`: its estimate of the total future *discounted* reward if the current policy continues. It does not directly steer. Its estimate gives the actor a point of comparison. A corner that earns some road reward might still be worse than expected if it leads to a crash just afterward. The critic learns by comparing its prediction with a return target estimated from collected rewards and later value estimates.

**A rollout.** The current actor drives for a fixed number of steps. For each step, PPO keeps the observation, sampled action, reward, the action's log probability under the **old** policy, the critic's value estimate, and whether the episode ended. A new track or a new episode starts when needed. This batch is the evidence for the next learning update. PPO can make several passes over that batch, then it collects new driving data; it does not keep using an unlimited replay buffer of old drives.

**Discounted return.** A reward received later usually counts a little less than one received now. The discount factor `gamma` controls this: a simple return would be `reward now + gamma × reward next + gamma² × reward after that + ...`. With `gamma = 0`, the agent would care only about the immediate step. A value close to `1` makes future outcomes matter more, which is important when slowing down for a corner prevents a later crash.

**Advantage.** The advantage asks, “Was this action better or worse than the critic expected from this observation?” A positive advantage tells the actor to make that sampled action more likely; a negative one tells it to make it less likely. As a rough example, if the critic expected a future return of 20 and the estimated return after an action is 25, the advantage is about +5. This is an estimate, not a known correct driving label.

**How PPO estimates advantage.** The common method is Generalized Advantage Estimation, or GAE. At one step it calculates a prediction error: `delta = reward + gamma × value(next observation) - value(current observation)`. It then combines this error with errors from following steps. The `lambda` setting controls how much of those later errors are included: smaller values rely more on the critic's estimates; larger values use a longer stretch of actual experience, which can be noisier. At a true episode end there is no future value to add. At a time-limit cut-off, the agent may still need to estimate future value. The training library handles these details, but I should understand what it is doing.

**The PPO update and clipping.** PPO compares the probability of each recorded action under the new policy with its probability under the old policy. The ratio is `new probability / old probability`. If an action had a positive advantage, increasing its probability helps; if it had a negative advantage, decreasing its probability helps. The clipped objective limits the *benefit in the training objective* of pushing that ratio too far. For example, with a clip range of 0.2, a ratio of 1.4 for a good action does not give more objective benefit than roughly 1.2. Clipping concerns probability ratios; it does not clip the steering wheel itself, guarantee a small total network change, or guarantee safe driving. [PPO paper](https://arxiv.org/abs/1707.06347).

**How the networks actually learn.** PPO splits a rollout into small batches and runs several training passes. The actor loss uses advantages and the clipped probability comparison. The critic loss measures the difference between predicted values and return targets. An entropy term can encourage the actor to keep trying varied actions instead of becoming too predictable too early. PyTorch computes gradients through these losses, and the optimizer changes the weights. If the CNN is shared, learning can change its filters as well as the actor and critic heads. I need to monitor whether the updates improve driving on new tracks, because lower training losses alone do not prove a better driver.

**The full loop in my words:** collect fresh drives → estimate returns and advantages → update actor and critic for a few passes → collect fresh drives again → evaluate without training noise. The critic helps the actor learn, but only the actor's actions control the car. My competition score comes from finishing laps quickly and avoiding DNFs, so I must judge checkpoints by those results as well as by training reward.
