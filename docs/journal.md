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

## Running the car demo and the Q-learning example

I learned how to run the CarRacing environment and the small Q-learning program from the project folder. I activate the virtual environment first, then run the script I want:

```bash
cd /home/dahmane/dev/max-verstappen
source .venv/bin/activate
python src/enviroment.py
```

The car window shows Gymnasium's `CarRacing-v3` environment. The script prints the image shape and pixel type, then takes 1,000 randomly sampled actions so I can watch the game. It resets if an episode ends and closes the environment when the loop finishes. This is only a way to see and interact with the game; the random actions do not mean the car is learning, and this script does not use a CNN or PPO yet.

The separate Q-learning example runs with:

```bash
python src/q_learning.py
```

That program trains a tabular agent in Gymnasium's `Blackjack-v1`, not in CarRacing. Its progress display shows completed hands and updates a few learning summaries, including the exploration rate (`epsilon`), recent win rate, and number of observation/action situations stored in its Q table. After training, it evaluates the learned policy over 1,000 hands without random exploration. This helped me practice the Q-learning idea on a small problem, but it is not a Q-learning driver for the car: Blackjack has a small discrete state and action space, while CarRacing gives image observations and continuous steering and pedal controls.

The two examples taught me different parts of the project: the car script shows how to start, observe, control, reset and close the environment; the Blackjack script shows a simple agent learning action values and displaying its training progress. My next step is to build and understand the CarRacing agent separately, one function at a time.

## Setting up PPO and planning the training

With coding assistance, I initialized the empty PPO files for the configuration, environment, rollout, GAE, loss, update and training loop. I also made separate folders for the basic CNN and the IMPALA CNN, plus a blank notebook for each. We have only created the structure so far. The functions are not implemented and I have not trained a PPO car yet. I want to fill the files one function at a time so I can understand what each part does.

I made a [training plan](TRAINING_STRATEGY.md) with the goal of winning on the unseen track too. I learned that I should train one driver on many different tracks instead of making a driver for each seed. The idea is to learn how to handle different roads and corners rather than depend on the three known tracks.

I will keep training seeds separate from validation seeds (`10000–10019`) and final test seeds (`20000–20019`). Validation will help me choose the model. The final test will stay untouched until I choose the settings. The known competition seeds, `202`, `42` and `203`, will also stay out of training. The mystery track is unknown, so I still need to see how well this plan transfers to new tracks.

I will start with PPO and the basic CNN, then compare the IMPALA CNN if the first driver is working. I want the fastest car, but I learned that finishing consistently matters too: a DNF costs 70 seconds. I will compare average penalized time and completion rate, not just the best lap or the training reward.

I also learned what `gae.py` is for. GAE estimates whether an action did better or worse than the critic expected, using rewards and value predictions from the rollout. PPO uses those advantages when updating the policy. When we implement it, we need to handle real episode endings and time limits correctly and avoid mixing the end of one race with the start of another.

## My plan to reduce overfitting

I want the car to learn how to drive on different roads, not just get good at the tracks it practices. Overfitting would mean it performs well on training tracks but struggles on new ones. Training reward alone will not tell me whether it can handle the mystery track.

For the first experiment, I will use 100 training tracks, with seeds `1000–1099`. The numbers are just identifiers: a higher seed does not mean a harder track. This range gives me 100 tracks without overlapping my evaluation groups. I understand that 100 is a starting choice, not a proven best number or a guarantee against overfitting.

At each new training episode, I will randomly choose from that pool. I will keep one driver learning across the different tracks instead of training a separate driver for each one. A track can appear many times, so 100 tracks does not mean only 100 episodes.

I will keep seeds `202`, `42` and `203` out of training for this baseline. Training on them is allowed by the challenge, but I want to first see whether the driver can handle them without practicing them. My validation tracks (`10000–10019`) and final test tracks (`20000–20019`) will also stay out of training. I will use validation to choose checkpoints, and only use the final test after choosing the model and settings. I should also avoid repeatedly changing the model just to improve the three public tracks.

During training, I will compare results on a fixed sample of training tracks with results on validation tracks, without learning or exploration noise during either evaluation. If training results improve while validation results stay poor or get worse, that is a warning of overfitting. I will compare completion rate and average DNF-penalized time, not just the fastest successful lap.

If that gap appears, I will inspect the driving and test a larger training pool or fresh random training seeds, still excluding every reserved evaluation seed. I will change one thing at a time and check whether it improves validation results. I will keep the best validated checkpoint rather than assume the last checkpoint is best. These steps help me detect and reduce overfitting; I still need actual experiments to know whether the car generalizes.

## Why I chose PPO for this environment

After studying PPO, I understand how its main parts fit together. I chose it as my first CarRacing algorithm because it can learn a policy for continuous controls and gives me an actor-critic training loop I can explain and build step by step. My action is three numbers: steering in `[-1, 1]`, gas in `[0, 1]` and brake in `[0, 1]`. I do not need to turn these controls into a small list of fixed driving actions.

PPO is my starting choice, not a claim that it is the best algorithm for this race. Its clipped objective discourages some overly large changes in action probabilities, but it does not guarantee stable learning or a winning car. It is on-policy: it collects fresh experience with the current policy, uses that rollout for several optimization passes, then collects new experience. It does not reuse an unlimited history of old drives like an off-policy method with a replay buffer. This can make collecting enough experience expensive, so I need to measure what fits in Colab. My reason for starting here is having a clear baseline I understand before comparing more methods. [PPO paper](https://arxiv.org/abs/1707.06347).

### From the track image to an action

My environment currently returns an RGB `uint8` image with shape `(96, 96, 3)`. It does not give the policy the track seed, car coordinates or a correct steering label. The seed stored in `info` is only for logging. The raw image is an observation, not the complete simulator state.

My planned preprocessing will resize the image, turn it into grayscale and stack four recent frames. A batch will have shape `(batch_size, 4, 84, 84)`. The recent frames can provide clues about motion that one image cannot show. Pixel values will be converted to floating point and divided by 255 once. At every new episode, the frame history must be reset so images from two different tracks do not get mixed together. This preprocessing is still a plan; `environment.py` currently returns the original RGB image.

The CNN will turn those images into features. The actor will use the features to produce a Gaussian action distribution, with a mean for each control and learned standard deviations. During training, sampling from that distribution provides exploration. The critic will use features to predict one value: the expected future discounted reward under the current policy. It does not predict steering. I plan to share the CNN between the actor and critic, with separate output heads, so both learning signals can update the image features.

For the initial Gaussian policy, I plan to store the sampled action before clipping, together with its log probability, and clip a copy to the environment's action bounds before stepping the game. PPO must compare old and new log probabilities for the same original sample. Replacing that stored sample with the clipped action would make the probability calculation inconsistent. Evaluation will use the policy mean with the same bounds. I must not introduce a different action transformation only when evaluating.

### What the car learns from

After an action, `env.step(action)` returns the next observation, reward, `terminated`, `truncated` and `info`. The default reward encourages visiting new road tiles and penalizes time; it is not a direct label saying which steering angle was correct. I will keep this reward for the baseline and measure actual finishes and lap times separately.

For each rollout transition, I need the observation, original sampled action, reward, old log probability, critic value, and episode-ending information. I also need the correct next-observation value for calculating targets. I will collect a fixed number of transitions, so a rollout may end halfway through a race or contain several races. I do not need a full completed lap before PPO can update.

### How the rollout becomes a learning signal

The critic estimates the future; GAE combines its prediction errors with the collected rewards to estimate advantages. A positive advantage encourages the sampled action, while a negative advantage discourages it. This is an estimate from experience, not proof that one action alone caused the outcome. With GAE, the critic's return target is the estimated advantage plus the stored value prediction.

I need to distinguish three boundaries. At a true termination, the future value is zero. At a time-limit truncation, I normally bootstrap from the final observation's value, but I stop the advantage recursion across the episode boundary. At the end of a rollout that stops during an ongoing race, I also bootstrap from the next observation. I must never use the reset image of a new race as the final image of the previous race.

### What the PPO update changes

For each stored action, PPO calculates `ratio = exp(new_log_probability - old_log_probability)`. Its actor objective uses the smaller of `ratio × advantage` and `clipped_ratio × advantage`, then averages over the minibatch. We minimize the negative of that objective. The clipping limits the incentive to keep pushing a probability ratio beyond the chosen range; it does not directly limit the car's steering or guarantee that every policy change is small.

The critic loss compares its current value predictions with the fixed return targets. An entropy bonus can encourage exploration. The combined loss lets PyTorch compute gradients, and the optimizer changes the network weights. Rollout values, old log probabilities, advantages and return targets must be treated as fixed data during these updates. I will shuffle the rollout into minibatches and make a limited number of passes before collecting fresh experience.

Lower losses alone will not convince me that the driver improved. I will also inspect action behavior, estimated policy change, value prediction quality and videos, then check completion rate and penalized time on validation tracks. The goal is fast, reliable driving on unseen roads.

## How I will implement and check each file

I want to implement one file at a time, explain its functions, run a small check, and discuss the output before moving on. I will record what actually happened, including failures, rather than call a planned feature finished.

1. **`environment.py`:** the first implementation is in place. It creates the game, samples training tracks and resets evaluation tracks. The initial checks passed for separate seed groups, reproducible sampling, observation shape and dtype, a valid environment step, and fixed-track resets. This checks the environment plumbing, not learning.
2. **`config.py`:** define the experiment settings and explain each one. Check that the rollout size and minibatch size fit together and that seeds and budgets are explicit.
3. **`rollout.py`:** define how transitions are stored and collected. Initially use a small temporary policy to check data shapes and reset handling; this will not demonstrate a trained driver.
4. **`gae.py`:** calculate advantages and return targets. Compare a short hand-calculated example with the output, including terminal, truncated and unfinished rollouts.
5. **`loss.py`:** implement the actor objective, critic loss and entropy term. Use small example tensors to inspect the probability ratio and clipping behavior.
6. **`update.py`:** combine minibatches, backpropagation and optimizer steps. Check that gradients are finite, weights change, and stored rollout targets stay fixed.
7. **`models/cnn/network.py`:** build the image encoder and the actor/critic interface needed by the training loop. Trace the shapes through the layers and check action and value outputs before connecting it to the game.
8. **`train.py`:** connect the checked parts, logging and checkpoints. Start with a short integration run, inspect the results, and only then plan a longer Colab run.
9. **`models/impala_cnn/network.py`:** add the residual image encoder after the baseline works. Compare it while keeping the PPO setup and evaluation protocol consistent.

The notebooks will follow these steps and show the observations and results. Implementing PPO's numerical pieces with small examples first will let me study them before adding the CNN, but meaningful image-based training needs the network connected to the complete loop. The remaining files are still empty at this point.

## First driving failure: the car kept going in circles

After the first basic-CNN training segment, I watched the driver on validation track `10000`. Instead of following the road, it kept looping around on the grass and did not make progress around the track. I first thought this was a bug because the training cell had finished successfully. I now need to separate a program that runs from a policy that has actually learned to drive.

The training log reported 50,176 total policy transitions. The longer segment added 49,152 transitions in 17.8 minutes, at about 46 transitions per second. The latest checkpoint was saved in Drive. Recent average training reward was around -229. These numbers show that training ran, but they do not show that the car can complete a lap.

The recorded validation attempt returned:

```text
Track seed: 10000
Finished: False
Raw simulator steps: 3000
Lap time: None
DNF-penalized time: 70 seconds
Reward: approximately -280.77
```

This was a failure to finish within the time limit. The video cell recorded the episode and displayed the failure; the result does not by itself show a video playback bug. I have not confirmed a code bug or a single cause for the circling.

Reviewing frames from the saved video confirmed that the car moves: within the first few seconds it is off the road, and later frames show repeated circular tire marks on the grass. The road changes orientation while the car stays near the camera center, so it is not simply one frozen picture. The roughly 60-second video is consistent with reaching the episode time limit. I still need the actual action values to explain the steering and pedal behavior.

One possible explanation is that the policy has not learned useful controls yet. The first 50,000 transitions were a benchmark, not a sufficient-training guarantee. With long episodes and 100 possible training tracks, this does not even mean the driver practiced every track. The training log also showed an action standard deviation near 0.985, meaning the Gaussian policy was still sampling broadly during training. Deterministic evaluation uses its means, so its behavior can differ from noisy training behavior. This is a diagnostic clue, not proof that exploration is the cause.

### How I will investigate and try to fix it

1. **Check which policy I watched.** Print the timestep counts of `model` and `evaluation_model`, and compare the latest checkpoint with any selected checkpoint. A previously loaded evaluation model can be older than the current training model. For diagnosis, explicitly evaluate the current `model` so I know which weights I am inspecting.
2. **Inspect the controls over time.** On a separate evaluation environment, log deterministic steering, gas and brake values while stepping the game. Check for non-finite values, persistent extreme steering, little gas, or simultaneous gas and brake. These could explain the visible behavior, but I need the actual outputs before claiming that they happened.
3. **Recheck the image pipeline.** Confirm four channel-first grayscale frames, uint8 before SB3 normalization, cleared frame history at reset, and identical training/evaluation action repetition. The earlier shape and reload checks passed, but those checks alone do not prove the policy learned anything useful.
4. **Compare familiar and held-out tracks.** Evaluate a small fixed sample from each group without learning. If both are poor, I should investigate basic learning and controls first. If training tracks are consistently much better, I should investigate generalization and training diversity. One failed validation track cannot establish overfitting.
5. **Continue only after those checks.** Preserve the 50,176-transition checkpoint, then, if the pipeline looks correct, continue the same baseline in measured chunks toward a cumulative 250,000 transitions. This is an experiment to see whether driving improves, not a confirmed fix. At the measured speed, the additional training would take roughly 72 minutes before extra evaluation overhead. I will inspect intermediate checkpoints and stop to investigate if behavior and reward do not improve.
6. **Change one thing if the failure persists.** Use the diagnostics to choose a controlled experiment involving action-distribution initialization, exploration or PPO settings. An action-distribution change may require a fresh run. I should not force gas or steering only during evaluation, because that would change the driver being measured. I will not jump straight to the larger CNN or claim that more layers solve the problem.

The issue is recorded, but it is not fixed yet. I need a new video and held-out completion/time results before saying a change worked.

### Checking the saved policy's actions

I loaded `latest.zip` and confirmed that it contained 50,176 training transitions. I then ran deterministic predictions on validation track `10000` in a separate environment, without training or changing the checkpoint. These were the sampled controls:

```text
Decision 0:   steering=0.279, gas=0.083, brake=0.000
Decision 40:  steering=0.645, gas=0.265, brake=0.000
Decision 80:  steering=0.676, gas=0.323, brake=0.000
Decision 120: steering=0.695, gas=0.207, brake=0.000
Decision 160: steering=0.543, gas=0.275, brake=0.000
```

At the sampled moments, the car was receiving gas and no brake. So these samples do not support my earlier suspicion that it was failing to move because it was braking while accelerating. Steering stayed positive and was fairly strong after the first sample, which is consistent with the repeated turning visible in the video. The policy was producing valid controls, but valid controls are not necessarily useful driving decisions.

These are five samples from the first 200 decisions, not every action or the whole race. They do not prove why the policy learned this behavior, whether it acts the same way on other tracks, or whether the full training pipeline is correct. I have identified a behavior to investigate, not a confirmed fix. Next I will compare a small sample of training and validation tracks, then decide whether to continue the baseline or investigate its learning setup further.

### Investigating exploration and the evaluation gap

With assistance, I reviewed the notebook's wrapper order and the installed SB3 rollout code. I did not find an obvious mismatch in image preprocessing, Gaussian action clipping or time-limit bootstrapping. This code review does not establish that there are no bugs. SB3's default initial log standard deviation is zero, which means a standard deviation of one. My log's value near 0.985 shows that exploration was still broad after the first segment.

Gaussian exploration samples continuous actions around the actor's means. SB3 clips the sampled controls sent to the game while retaining the original samples for PPO's probability calculation. My video used deterministic predictions, so the circling in that video came from the learned mean actions, not fresh exploration noise. Exploration could still have affected which experiences the policy learned from, but that connection is a hypothesis to test.

A diagnostic is now available to compare three training-pool tracks and three validation tracks with deterministic and sampled actions. It records raw means, standard deviations and expected clipping probabilities, saves a JSON report, and does not update the weights. The helper passed local integration checks on a short-check model with shortened episodes, including unchanged weights and restored Torch random state. Those checks validate the diagnostic plumbing, not the cause of my trained car's failure. My actual 50,176-transition checkpoint still needs to be tested with it.

If the pipeline checks pass, continuing the unchanged baseline gives one comparison. A separate possible experiment is starting a fresh run with `log_std_init=-0.5`, which starts exploration at a standard deviation of about 0.607 instead of one, while holding other settings fixed. This is not an established fix. Changing an initialization setting while loading old weights does not change the standard deviation already learned in that checkpoint. I will preserve the current run and decide from measured results rather than rewrite controls during evaluation.

## Inspecting my actual saved checkpoint

I shared the 50,176-transition checkpoint for inspection. It contained 36 completed episodes, totaling 49,236 decision transitions, plus transitions from episodes still in progress. So the first training segment could not have completed an attempt on every one of the 100 training tracks. The episode history does not contain the track seeds, so I cannot reconstruct its exact track coverage. The 245 recorded PPO epoch updates are not 245 completed races.

There was a separate portability problem when loading the notebook's serialized CNN from Python 3.13 into a Python 3.12 inspection environment. Reconstructing the reviewed CNN locally and loading its tensor state with strict checks avoided that problem. Its predicted controls matched the values I saw in Colab to rounding. This compatibility issue does not explain the circling inside my original Colab session.

The learned Gaussian standard deviations were approximately `[1.013, 0.980, 0.973]` for steering, gas and brake. The weights were finite and the input was normalized once to the expected range. On validation track `10000`, the raw initial means were approximately `[0.279, 0.083, -0.612]`; the negative brake mean is clipped to zero when sent to the game. A displayed brake action of zero does not mean that the Gaussian's raw brake mean is zero.

### Reproducing the deterministic driving failure

The policy was evaluated for a full episode on two training-pool tracks and two validation tracks:

- Seed `1000`: DNF at 3,000 raw steps, reward about -262.46, mean steering about +0.669.
- Seed `1001`: DNF at 3,000 raw steps, reward about -280.77, mean steering about +0.649.
- Seed `10000`: DNF at 3,000 raw steps, reward about -280.77, mean steering about +0.649.
- Seed `10001`: DNF at 3,000 raw steps, reward about -281.82, mean steering about +0.652.

Steering stayed positive on every decision in all four attempts. The last 100 decisions in each earned only the accumulated time penalty, about -20, without new road-coverage reward. This reproduces the circling behavior and shows a failure on training-pool tracks as well as held-out tracks. It is not evidence that overfitting to successful training driving is the main problem. Pool membership still does not prove those particular tracks were previously visited.

Along the deterministic validation trajectory, the Gaussian would put roughly 42% of steering samples and 62% of gas samples outside their legal ranges. Those are conditional probabilities for hypothetical exploration samples along that trajectory, not noise used in the deterministic video. Broad, heavily clipped exploration is a plausible learning difficulty, but these measurements do not prove it is the only cause. The confirmed behavior is a poorly learned policy that persistently turns one way after a small training budget.

### A checkpoint-selection fix and a recovery experiment

I found a separate issue in the checkpoint-selection logic: when every validation track fails, all checkpoints get a mean penalized time of 70 seconds. Keeping the first checkpoint on every tie could hide later progress. The notebooks now break equal-time ties using higher time-adjusted raw reward (`reward + 0.1 × raw_steps`), while keeping penalized time as the main score. Raw reward alone can favor an early crash because it accumulates fewer time penalties. Adding the original time penalty back gives a progress proxy while retaining any off-map penalty; it is not an exact tile count. Checks passed for better and worse tied-DNF checkpoints, rejecting an early-crash raw-reward advantage, exact ties, prioritizing better lap-time scores, and restoring the ranking from history. This did not cause the first failed video, because no best-validation checkpoint existed at that point.

A separate recovery notebook keeps the same CNN, image pipeline and track groups but tries gSDE, noise resampling every four decisions, noise-weight initialization at -2, and no entropy bonus. The [SB3 maintainers' CarRacing configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) motivates those exploration choices. Their full recipe also changes other settings and uses four million transitions, so this is not a reproduction of their results. The package of exploration changes is a practical experiment, not a single-variable causal test.

The recovery notebook uses a new folder and starts from scratch, preserving my first checkpoint. Its short local check passed collection, a PPO update, deterministic inference, saving/reloading, and a shortened evaluation. That only verifies that it runs. It has not yet demonstrated better driving or fixed the circling. I need to train it on Colab and compare held-out completion and time results before making that claim. In gSDE, the -2 initialization describes noise weights over policy features, not a fixed action standard deviation of `exp(-2)`.

### Checking whether sampled evaluation solves it

The same saved policy was also tested with sampled actions on seeds `1000`, `1001`, `10000` and `10001`. All four attempts still reached 3,000 raw steps without finishing. Rewards were approximately -187.37, -280.77, -251.92 and -233.33, respectively. Sampling sometimes earned more reward than the mean-action policy, but it did not produce a successful driver. One rollout per track is a diagnostic, not a reliable comparison of expected performance.

Across the four deterministic and four sampled attempts, there were zero finishes. Switching evaluation to random actions is therefore not an observed fix. The full measurements are saved in [the checkpoint diagnostic results](../notebooks/results/baseline_50176_diagnostics.json). My next test is longer training of the recovery experiment, compared against this preserved failed baseline, rather than saying the issue is solved because a new notebook runs.

## What I understand now and my next plan

This is my explanation after working through the questions and debugging with assistance. I want to keep the concepts separate so I can explain my own driver instead of only knowing which cells to run. My updated [training strategy](TRAINING_STRATEGY.md) records the experiments. The first recovery chunk reached 25,600 transitions. The later results through the confirmed 225,424-transition backup, interruptions and IMPALA update are recorded in [my latest learning entry](#what-i-learned-from-the-later-runs-and-what-i-am-changing).

### The agent, its observations, actions and rewards

My agent is the learned driver. The environment is Gymnasium CarRacing. The simulator has a full state containing the car position, speed, wheel dynamics and road layout, but my policy does not receive those hidden numbers. Its observation comes from the image.

The raw image is `(96, 96, 3)` RGB with uint8 pixel values. My preprocessing makes grayscale images of size `84 × 84` and stacks four recent decision-time frames. For one observation, the shape is `(4, 84, 84)`; a batch of two is `(2, 4, 84, 84)`. Recent frames provide motion clues. Missing history is zero-padded at reset, and the previous episode's images must not leak into the next episode. SB3 converts pixels to floats and divides by 255 once.

The action is `[steering, gas, brake]`. Steering ranges from -1 to 1, and the two pedals range from 0 to 1. For example, `[0.1, 0.5, 0.0]` is a small steering input, half gas and no brake. The actor has to learn useful values from experience; the environment does not supply the correct action as a label.

I also learned that reward does not simply add one for every correct-looking movement. The original environment subtracts 0.1 per simulator step and gives `1000 / number_of_track_tiles` when a road tile is first visited during the episode. If an imaginary track has 250 tiles, one new tile gives 4 before that step's time penalty, so one such step gives 3.9. Revisiting that tile does not pay again. Leaving the playfield gives a -100 step reward and ends the episode.

With action repeat 2, one policy decision normally executes two simulator steps and adds their rewards. Two steps without new tiles give about -0.2. The wrapper stops immediately if an episode ends. Grass has no additional grass penalty in this baseline. The 70-second DNF score belongs to evaluation, not the reward fed to PPO. Circling on grass can therefore keep accumulating time penalties without earning new road rewards.

### What the CNN outputs and why it is connected to the actor

The CNN outputs 256 learned feature values for each observation. These are not action probabilities, steering values, rewards or a prediction of the next image. They are a representation that the rest of the network can use. I cannot assume one feature means speed and another means a corner; useful information can be spread across many numbers.

In the basic CNN, the image goes through three convolutions:

```text
4 × 84 × 84
→ 32 × 20 × 20
→ 64 × 9 × 9
→ 64 × 7 × 7
→ flatten to 3,136 values
→ linear layer and ReLU
→ 256 features
```

The three-layer layout is a compact starting baseline, not a proven best depth for this race. The filters are learned rather than written as road-detection rules. The actor takes the 256 features through a 64-unit hidden layer and produces three means. The critic takes the same shared features through its own 64-unit hidden layer and produces one estimated future-return value.

```text
Recent images → shared CNN → features → actor head → action distribution
                                    ↘ critic head → value estimate
```

The actor is the action-producing path, including the image features it depends on. PPO is the training algorithm, not another neural network receiving the CNN features. A policy maps an observation to an action or action distribution. A value function estimates future discounted reward under a policy. A Q-function evaluates an observation/state and a particular action. These are different jobs, and none is automatically a prediction of the next game state.

### Gaussian sampling, exploration and clipping

I was mixing up several operations. The terms I need here are Gaussian sampling, exploration, exploitation and clipping. I should use those names rather than treat "crossing" as a separate component of PPO.

A deterministic policy gives one action for an observation. Our original PPO policy is stochastic during training: it describes a Gaussian distribution of possible continuous actions, then samples one. This lets the agent try alternatives instead of always repeating its current preference.

The mean is the center of that distribution. It is calculated from the observation by the actor. It is the policy's current preference, not a known correct answer. In the original diagonal-Gaussian setup, three separately learned log-standard-deviation parameters provide the spread for the three controls. Those spread parameters are shared across observations, while the means depend on the images.

Standard deviation describes how widely samples vary around a mean. It is not an accuracy score, a probability, an exploration percentage or a calibrated statement of confidence. The computer supplies a random draw from a standard Gaussian. The original sampling equation for one control is:

```text
sampled action = mean + standard deviation × random draw
```

For example, with mean 0.2, standard deviation 0.1 and random draw -0.5, the sampled action is `0.2 + 0.1 × (-0.5) = 0.15`. The draw itself is not a learned parameter. Training changes the weights that produce the means and the parameters that control the spread.

A standard deviation near 1 is broad relative to our action ranges. It is not automatically wrong, but many requested controls may lie outside the legal bounds. SB3 clips the copy sent to the game. It keeps the original sampled action and its old log probability for PPO. Replacing the stored sample with its clipped copy would make that probability calculation inconsistent. This action clipping is separate from PPO's probability-ratio clipping and from gradient clipping.

Making exploration tiny would not repair a wrong mean. If the mean keeps steering around +0.65, narrowly repeating it can still produce circles. My deterministic video already used the mean with exploration noise disabled. Exploration during training may have affected what the policy learned, but random noise was not being added to the actions shown in that deterministic video.

### Why discrete actions use a different distribution

If an agent chooses exactly one of left, right, up and down, the policy can produce four logits. Softmax converts those scores into probabilities that sum to one, and a categorical distribution selects a direction. It needs randomness for sampling during training, but not a Gaussian mean and standard deviation. Simultaneous button presses would need an action representation that allows combinations.

The car chooses continuous numbers, so I cannot give a separate output for every possible steering angle. A Gaussian is a convenient distribution described by a few parameters. It is a design choice; not every continuous policy must be Gaussian.

For discrete actions, the distribution gives the probability of a selected action. For continuous actions, it gives a probability density. The probability of one exact continuous value is zero; probabilities apply to intervals. Density is not a percentage and can exceed one. The `log_prob` name in the library is used for the log density in this case.

For the original three independent Gaussian controls, the joint log density is the sum of the three individual log densities. The distribution supplies this log likelihood; the loss does not create it. PPO later compares the old and new likelihoods of the same recorded action. In the original setup, log likelihoods refer to the original sample before action clipping.

### How PPO uses experience to learn

There is a collection stage and an update stage. During collection, the actor chooses an action, the critic predicts a value, and the game returns the reward and next observation. The rollout stores the observations, sampled actions, rewards, old log probabilities, values and episode-ending information.

GAE combines rewards and critic prediction errors to estimate advantages. A positive advantage says an action's estimated outcome was better than the critic expected; a negative one says worse. It does not prove that the action alone caused the outcome. The return target for the critic can be formed as the stored value plus the estimated advantage.

For an ongoing episode, the calculation starts from `delta = reward + gamma × next_value - current_value` and combines later errors using `gamma × lambda`. Our starting values are `gamma=0.99` and `lambda=0.95`, applied per collected decision transition. As a simple illustration, a stored value of 10 and estimated return of 12 give an advantage of +2. The return target is still an estimate, not a known correct future.

True termination removes the future value. A time-limit truncation normally still bootstraps from the final observation's value, but the advantage recursion must not cross into the next episode. An unfinished rollout also needs a next-value estimate. The auto-reset image of a new race is not the final image of the previous race.

During updates, PPO recalculates the recorded actions' log probabilities under the current policy and forms `exp(new_log_prob - old_log_prob)`. Its clipped objective uses that ratio and the advantages to encourage or discourage the actions without giving unlimited benefit for pushing certain ratios farther. PPO makes several minibatch passes over the fresh rollout, then collects new experience. It does not keep an unlimited replay history of old drives.

The rollout's old log probabilities, values, advantages and return targets are fixed data during this update. The actor loss must not backpropagate through a changing advantage target into the critic. The critic has its own value-prediction loss.

### Which losses update the shared CNN

I initially thought the actor and critic losses only affected their separate heads. I now understand that both heads depend on the CNN, so both losses can send gradients into its shared weights.

The total objective in our implementation can be written as:

```text
loss = actor loss + vf_coef × critic loss − ent_coef × entropy
```

`vf_coef=0.5` is a chosen weighting coefficient and the SB3 default we kept. It is not the Gaussian standard deviation, a probability or a value calculated from an image. Before any gradient clipping, a shared CNN weight receives the actor gradient plus 0.5 times the critic gradient, along with any entropy gradient that actually depends on that weight.

The actor mean head receives the actor-objective gradient. The critic head receives the value-loss gradient. The exploration parameters receive gradients through their role in the action distribution. In the original diagonal Gaussian, entropy depends on the learned standard deviations rather than the means, so its entropy term does not directly teach the CNN through the mean head. With gSDE, the effective variance can depend on features; the recovery entropy coefficient is zero anyway.

PyTorch calculates these gradients from the computation graph:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The loss is calculated; the weights are updated. I do not update a stored "loss parameter." The separate `max_grad_norm=0.5` setting limits the combined gradient norm before the optimizer step. It happens to have the same numerical value as `vf_coef`, but it has a different purpose.

The shared CNN gets no labels saying what its features should mean. It changes because its features affect action likelihoods and value predictions. Sharing can save computation and learn features useful to both jobs, but actor and critic gradients can also conflict. Separate CNNs are another design, not something I have tested here.

### What the first run actually established

The first checkpoint reached 50,176 transitions and contained 36 completed episodes. Its deterministic video circled on the grass and timed out. Inspection reproduced zero finishes in four deterministic and four sampled attempts across two training-pool and two validation tracks. The learned steering stayed positive throughout the deterministic attempts. The weights were finite, and inspected predictions matched Colab.

This is evidence of a poorly learned turning policy. It does not prove that the CNN has too few layers, that the agent stopped exploring, or that successful training driving was simply overfit. The initial standard deviations remained broad. Switching evaluation to random actions did not produce a successful driver. A small training budget and poor exploration experience are plausible contributors, but I have not isolated one unique learning cause.

I also fixed a separate checkpoint-selection issue: all-DNF models tie at 70 seconds, so retaining the first model forever can hide progress. The updated tie-breaker uses time-adjusted raw reward while keeping penalized time primary. It is a selection fix, not a change to the reward used for training and not a repair of the actor by itself. The Python-version checkpoint-loading issue was another separate tooling problem, not the cause of circling in the original Colab run.

### What I am testing next

The recovery keeps the basic CNN and original environment reward, but changes exploration to gSDE, resamples its noise matrix every four decisions, initializes noise weights on a log scale at -2, and removes the extra entropy bonus. gSDE uses policy features with sampled noise weights, so exploration has more temporal structure while actions can still change with the observation. The -2 setting does not mean a negative standard deviation or a fixed action spread of `exp(-2)`. Zero entropy coefficient does not mean zero exploration.

These choices are motivated by the maintainers' CarRacing configuration, not by proof that they solve my run. Their full recipe differs in other settings and training budget. My recovery changes a package of exploration settings, so an improvement would not identify which one alone caused it. I will preserve the original checkpoint and use a separate folder. The gSDE policy's exploration parameters have different shapes, so I should not load the old diagonal-Gaussian checkpoint into that fresh run.

The short recovery integration checks passed, and the first recovery training chunk completed in Colab. Its statistics are recorded below. Later videos and validation results are recorded in the latest entry. I will not call the circling fixed just because the notebook runs.

### My training counts and next actions

A transition is one observation/action/reward interaction. An episode is a whole driving attempt. A track seed identifies a road. These are different counts. The 100 training seeds are sampled with replacement at episode resets; the agent does not automatically visit them in order.

Two environments each contribute 512 decisions to a 1,024-transition rollout. Minibatches contain 128 transitions, and PPO can make up to five shuffled passes, stopping earlier if the approximate-KL condition is triggered. A smaller overall training chunk is not a smaller minibatch. I will keep rollout and minibatch settings unchanged initially and use cumulative targets of 25,000, 50,000 and 100,000 to inspect progress, extending toward 250,000 and beyond only according to results and compute. Completing a rollout can slightly exceed the requested counter.

The guidelines require a maximum of 3,000 raw steps per race, not a particular total training count. At 50 raw steps per simulator second, the episode limit is 60 seconds. A DNF gets 70 seconds in scoring. The 17.8 minutes reported earlier was wall-clock training time. I should not confuse a finished training segment with a trained, competitive car.

For the current recovery run I will let an active cell finish, preserve the checkpoint, inspect the selected model and record its validation video. I will continue the same model between chunks rather than click Run all again. After a disconnect, I will rerun setup and definitions and load the recovery's `latest.zip`; saving the notebook alone does not save its RAM. Fresh runs and resumed runs need different loading choices.

I will keep training seeds `1000–1099` separate from validation `10000–10019`, final test `20000–20019`, and public tracks `202`, `42`, `203`. One hundred is a starting pool, not a guarantee against overfitting. I will compare training-pool and validation results, check actual exposure where possible, and broaden training diversity if the evidence supports it. The final test stays untouched until model selection.

I will select checkpoints primarily by mean penalized time and report completion rate and finishing lap times too. Before the first periodic validation at 100,000 transitions, there may only be a latest checkpoint. A file named best can still fail every track. I will check which timestep count is being evaluated and use the full validation set for serious comparisons. I will keep final testing off while tuning.

The IMPALA residual CNN remains a later comparison. It uses shortcut additions `x + F(x)` to help information and gradients pass through a deeper encoder. Its extra layers do not automatically solve the current learning failure. A fair encoder comparison must keep exploration, PPO, preprocessing, tracks and budget consistent; the original IMPALA notebook and the recovery basic CNN currently differ in exploration too.

To watch the selected trained model locally, I can download its checkpoint and use `src/watch_agent.py`. The `src/set_up.py` demo still samples random actions. The viewer does inference only; it does not train or magically improve the model. A video explains behavior, while multiple held-out races establish performance. Submission will still need the required PyTorch weights and a verified inference interface, not just the training zip.

My next evidence should be the recovery's action behavior, validation video and completion/time measurements. That is how I will decide whether to continue, revise the exploration setup or investigate another problem.

## The failures we found, the verified fixes and my recovery experiment

### A display error interrupted the run at 100,000 transitions

The next recovery log reached exactly 100,000 transitions, then raised `RecursionError: maximum recursion depth exceeded`. The traceback points to the validation callback's summary print, through Rich's stdout proxy and IPython's display/flush handling. This was a notebook-output failure during validation, not evidence that PPO weights exploded. The repeated validation lines came from recursive rendering, not thousands of independent evaluations.

The log reports that the `finally` block saved `latest.zip` and that the segment collected 48,800 transitions in 18.7 minutes. The validation summary had five tracks, zero finishes, a 70-second mean penalized time and mean reward about -125.23. These five-track results should not be directly equated with the earlier three-track mean. Execution failed before the callback's best-checkpoint selection code, so I should not assume this evaluation produced `best_validation.zip`.

The local notebooks now use `progress_bar=False` to avoid this Rich progress display path. Ordinary SB3 text logs and checkpointing remain enabled. This is a targeted mitigation based on the traceback; it still needs confirmation in the affected Colab runtime. I must make the same change in my already uploaded notebook. I should save a separate interruption backup, verify its timestep counter, and continue the same model toward the cumulative target. Some collected transitions at the end of an interrupted rollout may not have been used for a PPO update; the counter is not a count of completed optimization steps.

### The car circling was a real failure, but its learning cause is still uncertain

The original policy moved off the road and kept turning on the grass. It was not just a frozen video. The deterministic policy kept choosing positive steering, and all eight diagnostic attempts failed to finish. Randomly sampling from that same policy also failed. This means I cannot fix the result simply by turning randomness on when watching it.

The deterministic video did not add exploration noise. Its immediate problem was the learned mean action. Broad exploration during training and too little useful experience may have contributed to learning that mean, but neither has been isolated as the single cause. I found no evidence of NaN weights or an exploding-parameter failure in the inspected baseline. I should not call every poor learning result a confirmed code bug.

### A checkpoint-selection bug and a separate loading problem were fixed

When every candidate fails to finish, their primary scores are all 70 seconds. Keeping the first checkpoint on every tie can hide useful progress. The verified selection change keeps penalized time first, then uses higher `reward + 0.1 * raw_steps` to break equal-time ties. Adding back the time cost avoids favoring an early crash just because it accumulated fewer time penalties. This remains a progress proxy, not a measured lap completion percentage. Focused checks covered better and worse DNF candidates, exact ties, early-crash bias, the primary score and restored selection history.

The local loading problem came from a notebook-defined class saved under a different Python minor version. The viewer now rebuilds the architecture and loads the tensor weights. Its predictions matched the uploaded checkpoint, and short original-CNN, gSDE and IMPALA loading/video checks passed. A desktop window still needs checking on a machine with a display. This tooling fix does not change the learned actions or explain why the original Colab policy circled.

### What the recovery actually changes

The recovery keeps the same three-layer basic CNN, four stacked grayscale frames, actor and critic heads, track pools and original reward. It is still PPO. I started it as a fresh experiment in `basic_cnn/gsde_recovery_01`, preserving the failed baseline in `basic_cnn/run_01`.

The main changes are:

- `use_sde=True`: the exploration term now depends on policy features. These come from the images through the shared CNN and actor hidden layer. Instead of drawing a completely fresh independent action perturbation at every decision, gSDE samples noise weights and combines them with those features.
- `sde_sample_freq=4`: during collection, new noise weights are sampled every four decisions. The weights are reused between resamplings, but the images, features, actor mean and final action can still change. It does not force the car to repeat one action four times. With action repeat 2, this interval normally spans eight simulator steps per environment.
- `log_std_init=-2.0`: this initializes the noise-weight scale at approximately `exp(-2) = 0.135`. It is not a negative standard deviation and not a fixed action standard deviation of 0.135. The effective action spread also depends on the features.
- `ent_coef=0.0`: I removed the extra entropy bonus that rewards a broader action distribution. The policy still samples actions during training and can still learn its exploration scales through the PPO objective. Removing this bonus does not turn training into deterministic driving.

These choices follow part of the [SB3 maintainers' CarRacing configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml). Their complete configuration also differs in preprocessing, reward normalization, network settings and optimization, and specifies four million training transitions. I have borrowed an exploration setup, not reproduced their complete experiment or established that 25,000 transitions should be enough.

The actor and critic still train the shared CNN through their losses. With the entropy coefficient zero, our weighted loss is `actor_loss + 0.5 * critic_loss`. The exploration distribution changes how actions are sampled and how their likelihoods are evaluated; it does not replace the critic or give the CNN road labels.

### What my first recovery log shows

The checkpoint reached **25,600 total transitions**. The long segment collected **24,576 transitions in 8.6 minutes**, around **47.5 transitions per second**, after the initial 1,024-transition check. It saved `latest.zip` to `MyDrive/max-verstappen/basic_cnn/gsde_recovery_01/`.

The requested cumulative target was 25,000. After the initial check, 23,976 transitions remained. PPO finished complete 1,024-transition rollouts, so it collected 24,576 more and reached 25,600. These are totals across both environments, not episodes or a budget for each track.

The final displayed metrics were:

- Mean episode reward: about **-245**. The early displayed reward was about -258, but the intermediate values fluctuated. This is weak evidence of progress, not a clear successful learning curve. Different sampled tracks also affect this average.
- Mean episode length: about **1,400 policy decisions**. With action repeat 2, that is roughly 2,800 raw steps per episode. Staying active for a long time does not prove a lap was completed; a car can also spend that time circling.
- Approximate KL: **0.0126**, a measure of policy change on the collected data, not driving accuracy.
- Clip fraction: **0.101**. About 10.1% of evaluated probability ratios were outside PPO's clipping interval. It is not a crash rate or a percentage of clipped steering commands.
- Explained variance: **0.574**. The critic captured some variation in its return targets at that point. This does not mean 57.4% driving accuracy or 57.4% of a lap.
- Value loss: **2.51**, with earlier spikes including **80.2**, **53.2** and **59.1**. The critic's errors were uneven; a lower final value does not establish that learning is fully stable.
- Logged `std`: about **0.130**. Under gSDE this summarizes learned noise-weight scales, not the action standard deviation for every image. It cannot be compared directly with the original policy's three action standard deviations near one.

I also saw `Early stopping at step 0 ... max kl: 0.06`. I checked the installed PPO code: it stops the current optimization phase when a minibatch's approximate KL exceeds `1.5 * target_kl`. Our target is 0.03, so that threshold is 0.045. The printed step is the optimization epoch index, not the game step. PPO skipped remaining update passes for those rollouts and continued collecting experience. The completed checkpoint confirms this was not a stopped training job. Repeated warnings in future chunks would be a reason to inspect update size, not to remove the safeguard automatically.

The `entropy_loss` entry still appears because SB3 calculates and logs it even when its coefficient is zero. It contributes zero to our total loss. The ipywidgets deprecation warning concerns notebook display code; it did not stop this training segment.

At this first-chunk stage, I could not say the recovery fixed the car: that log had no recovery completion rate or driving video. I also could not fairly rank its -245 training reward against the baseline's roughly -229: they used different budgets, and training reward alone is not held-out race performance.

### The steps I planned after the first recovery chunk

1. Keep the current runtime and model. Print `model.num_timesteps` and confirm that `RUN_DIR / "latest.zip"` exists. The counter should be 25,600 for this checkpoint.
2. Run the model-selection cell in section 14 with `RUN_FULL_VALIDATION=False`. It still sets `evaluation_model`. Before the first periodic validation at 100,000 transitions, a best-validation checkpoint may not exist, so I should check that the current recovery model is selected.
3. Set `RECORD_VIDEO=True` in section 15 and run that cell. Read the printed `finished`, `raw_steps`, `reward` and time fields, then watch the behavior on validation seed 10000. A single video is a diagnostic, not proof of generalization.
4. Preserve this checkpoint and its result before continuing. In section 12, use `RUN_LONG_TRAINING=True` and `TARGET_TOTAL_TIMESTEPS=50_000`, then rerun only that training cell. The existing counter is subtracted automatically; this continues the same model rather than starting another 50,000 from zero. With complete rollouts, I expect approximately 50,176 total transitions.
5. Select the evaluation model again and record the next video. Continue toward a cumulative 100,000 if the measurements justify it. Keep the current minibatch and rollout settings for this comparison.
6. For a serious comparison, enable `RUN_FULL_VALIDATION=True` and evaluate all 20 validation tracks plus the five training-pool probes. Use completion rate and penalized time to judge progress. Leave `RUN_FINAL_TEST=False` while choosing settings.

If Colab disconnects, I need to rerun setup and definitions and explicitly resume the recovery's own `latest.zip`. I should not recreate a fresh model, load the original Gaussian checkpoint into this experiment, or assume the notebook file contains the trained weights. The [SB3 inference guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) also explains that stochastic gSDE inference needs explicit noise resets; our scored videos stay deterministic.

## What I learned from the later runs and what I am changing

### I thought the recovery would fix the circles, but I had to check the result

When I watched the recovery, I said it started and then kept turning. The first thing I needed to check was which model I was actually watching. A video can use `evaluation_model`, while training continues with `model`. If I do not select again after training, I can accidentally watch an older model.

At the 50k check, I confirmed that both counters were 50,176, the folder was `basic_cnn/gsde_recovery_01`, gSDE was enabled and the checkpoint existed. This confirmed it was the recovery model, not the old ordinary-Gaussian baseline. The videos still showed it leaving the road and circling on the grass. Starting the car is not the same as learning to drive a lap.

### My six-track check at 50,176 transitions

I evaluated the current model on three training-pool tracks and three validation tracks without changing its weights. The results were:

- Training seed 1000: reward **-122.5256**, no finish.
- Training seed 1001: reward **-203.8462**, no finish.
- Training seed 1002: reward **-220.0000**, no finish.
- Validation seed 10000: reward **-229.4872**, no finish.
- Validation seed 10001: reward **-230.3030**, no finish.
- Validation seed 10002: reward **-230.8176**, no finish.

Every attempt reached 3,000 raw steps and received 70 seconds as its DNF score. That is **zero finishes out of six**. These checks showed failure in both groups, not proof that the model learned training roads and only failed on unseen roads. Also, a seed being in the training pool does not prove the model had already practiced that particular seed.

Some new tiles were visited, so the rewards were not all the same. But road progress is not a completed lap, and six tracks are only a small diagnostic. I need finishes and lap times to judge whether it can compete.

### What I understood about increasing the number of steps

I kept asking whether to use 100k, 500k, a million or even four billion. I understand now that `TARGET_TOTAL_TIMESTEPS` is a cumulative training counter across the environments. It is not the length of a race, the number of tracks or the number of episodes.

If my model has 100,000 transitions and I set the target to 500,000, it asks for about 400,000 additional interactions. I keep `reset_num_timesteps=False`. I can jump straight to that larger target; I do not have to stop at 250k. The intermediate targets are inspection points. They do not reset the weights or introduce a special learning stage.

The race limit stays 3,000 raw simulator steps. A rollout is still 512 decisions from each of two environments, or 1,024 transitions before an update phase. The minibatch is still 128. Changing the overall target does not change any of those settings.

More transitions give PPO more experience and more opportunities to update the policy, but they do not guarantee that the latest model is the best. It can improve, stall or get worse. I should keep earlier checkpoints and select using validation. The maintainers' reference uses four **million**, not four **billion**, and its other settings differ from ours. At about 47.5 transitions per second, four billion would take around 2.7 years for one model before extra overhead. It was not a realistic deadline plan.

The time estimates were only estimates based on the basic CNN. At that earlier throughput, going from 100k to 500k would take roughly 2 hours 20 minutes plus evaluation overhead. IMPALA needs its own measured speed. I should not assume that a deeper model trains at the same rate.

### What really stopped the run at 100k

I saw a red cell status near a progress bar saying 100%, and I thought the training had broken. Reading the actual log was more useful than guessing from the screenshot.

The counter reached exactly **100,000**. The validation callback evaluated five tracks and produced zero finishes, a mean penalized time of 70 seconds and mean reward around **-125.2281**. Then printing that summary triggered `RecursionError: maximum recursion depth exceeded` through Rich's output proxy and Colab/IPython's display flushing. The repeating summary lines were the recursive output failure, not evidence of thousands of evaluations.

The `finally` block reported saving `latest.zip`. That segment collected 48,800 transitions in 18.7 minutes, with reported throughput around 43.4 per second. The error happened before the callback's best-checkpoint selection, so I could not assume that this validation had created `best_validation.zip`.

The local notebooks were changed to `progress_bar=False`, including the smoke check and the longer training call. The normal SB3 training tables remain available. This avoids the Rich progress display path implicated by the traceback. The updated cells passed syntax checks and short runs without that progress bar; I still need to distinguish that from reproducing a full 100k Colab validation successfully. An already uploaded notebook needs the same edit; changing a local file does not update my Colab copy.

This was separate from the earlier checkpoint-selection tie bug and Python-version loading issue. Fixing display, selection or loading does not by itself fix driving. Also, an interrupted rollout can contain counted transitions that have not yet been used in a PPO update.

### At 100k, it did not just circle: it drove out of the playfield

The short `wtf.mp4` clip lasted about 3.76 seconds. The car left the road, then reached the edge of the visible playfield. I checked the model metadata and episode result instead of assuming it was a shortened version of the old timeout:

- Current model: **100,000 transitions**.
- Evaluated model: **100,000 transitions**.
- gSDE: **True**.
- Validation seed: **10000**.
- Finish: **False**.
- Raw steps: **190**, or **3.8 simulator seconds**.
- Penalized time: **70 seconds**.
- Reward: **-54.7974**.

Together, the early ending and video are consistent with leaving the allowed playfield. The larger-looking reward was not proof of better driving. It stopped much earlier and accumulated fewer time penalties. Both the timeout and early failure get a DNF score of 70 seconds. I also cannot compare this one track directly with the five-track validation average as if they were the same measurement.

### I printed the actual controls to understand the failure

The deterministic rollout on seed 10000 gave these samples:

```text
Decision 0:  steering= 0.675, gas=0.738, brake=0.000
Decision 10: steering= 0.026, gas=0.700, brake=0.000
Decision 20: steering= 0.305, gas=0.482, brake=0.000
Decision 30: steering=-0.464, gas=0.502, brake=0.000
Decision 40: steering= 0.490, gas=0.526, brake=0.000
Decision 50: steering= 0.034, gas=0.772, brake=0.000
Decision 60: steering= 0.001, gas=0.792, brake=0.000
Decision 70: steering= 0.016, gas=0.788, brake=0.000
Decision 80: steering=-0.262, gas=0.978, brake=0.000
Decision 90: steering=-0.204, gas=0.762, brake=0.000
Episode ended. Raw steps: 190
```

This is different from the first baseline's persistent positive steering. Here the sampled steering values changed direction, gas stayed substantial and brake was zero in every printed sample. It does not prove that braking was zero between samples. It also does not prove that the changing steering was useful road following.

The video and controls suggest an accelerating policy that failed to stay on the road and recover. They do not isolate why PPO learned it. Zero braking alone is not a bug: the goal is finishing the race, not making every action output nonzero. I did not add hand-written braking or throttle rules to evaluation and call that a trained-model fix.

Because the probe was deterministic, gSDE noise was not being added to those actions. Exploration changes the experience during training; it does not explain the video as random noise being applied at evaluation. This is another reason to keep the policy's mean behavior separate from its exploration distribution.

### Restarting Colab caused a different error

After I restarted the runtime, the training cell raised `NameError: name 'model' is not defined`. The runtime had cleared its Python memory. That did not mean the saved Drive weights had disappeared.

To resume, I need to rerun installation/imports, mount Drive, keep the same configuration and experiment folder, and define the wrappers and encoder. Before running the loading cell, I set `RESUME_FROM = str(RUN_DIR / "latest.zip")`. Then I define the callback, skip the smoke update with `RUN_SMOKE=False` if it is unnecessary, and continue the cumulative target with `reset_num_timesteps=False` and `progress_bar=False`.

Loading restores saved policy weights, optimizer state and counter. It does not restore the exact simulator state or reproduce every interrupted random draw. Leaving `RESUME_FROM=""` would request a fresh model; the existing-run guard is there to stop accidental overwriting.

### My later GPU interruption and the checkpoint I actually saved

I reported that the next run stopped with a GPU error and described its progress as about 250k. I have not provided the exact GPU traceback, so I do not know whether it was a runtime allocation problem, a CUDA error, memory exhaustion or another GPU issue. I should not treat those as the same failure as the Rich recursion error.

The confirmed backup output was:

```text
Backup: /content/drive/MyDrive/max-verstappen/basic_cnn/gsde_recovery_01/backup_100000.zip
Transitions: 225424
```

So the model I actually confirmed saving had **225,424 transitions**, not 250,000. The filename came from the string I gave `save_checkpoint`; it did not change the contents back to 100k. Calling a file `backup_100000.zip` does not make its internal counter 100,000.

For future backups I will use `save_checkpoint(model, f"backup_{model.num_timesteps}")`. I was advised to save a correctly named copy, but I have not yet supplied confirmation that this second copy exists. The confirmed preserved file is the one above.

In `wow.mp4`, I saw the car drive along parts of the road and thought it was getting on track, although it was not the fastest. Inspection also showed it leaving the road and circling on the grass during the roughly 60-second clip. This suggests some more useful behavior than the immediate departure, but there is no supplied episode-result dictionary or verified checkpoint counter for that video. I cannot record it as a completed lap or a proven speed improvement.

### Why I am not training only on the three public tracks

I asked whether training on seeds 202, 42 and 203 before submission could make the car faster. It could specialize the policy to those roads, but it would not guarantee speed or performance on the mystery track.

The guideline's training advice says to train on random seeds and only use those three to test. I will follow that advice and keep the public seeds out of the training pool. This advice should not be rewritten as a separate formal prohibition that the document does not state. The competition includes an unseen track to reward general driving.

I can evaluate the public tracks using `evaluate_tracks(model, PUBLIC_SEEDS, split="public")` without updating the weights. I have not yet supplied those results. I will not claim I tested or finished the three public tracks just because I discussed doing it.

### What we changed in the IMPALA notebook

With the deadline close, I asked whether to train the best three models. The practical plan was at most two configurations: the existing basic CNN and one IMPALA comparison, keeping useful checkpoints rather than starting three unrelated experiments. Checkpoints are separate candidates, not an ensemble of three drivers.

We found that the old IMPALA notebook used ordinary Gaussian exploration with an entropy coefficient of 0.01. Comparing it directly with the gSDE basic CNN would change the exploration setup as well as the encoder. The IMPALA notebook is now updated to match the recovery's training configuration except for the encoder:

```python
use_sde = True
sde_sample_freq = 4
log_std_init = -2.0
ent_coef = 0.0
```

These settings are passed into the PPO constructor and policy arguments, not just written into `CONFIG`. The residual encoder is kept. PPO, grayscale 84-by-84 images, four stacked frames, action repeat 2, actor/critic heads, original reward and seed groups stay matched to the recovery configuration. Equal configuration does not mean both policies collect identical experience.

The new default folder is `impala_cnn/gsde_comparison_01`. It starts from fresh IMPALA weights, so I must not load the basic-CNN checkpoint into it. I also should not resume an older ordinary-Gaussian IMPALA run with these new settings. The first cumulative comparison target is 100,000, and long training remains off until explicitly enabled. If time permits, it needs a comparable training budget before an encoder comparison is meaningful.

The updated notebook keeps `progress_bar=False`, explains the gSDE interpretation of logged standard deviations, and replaces the old diagonal-Gaussian-only diagnostic with a deterministic control inspector. This inspector saves actions to JSON and uses a separate evaluation environment. Its default is only the first 200 decisions, so it is not a full race score.

The updated IMPALA notebook passed code syntax checks, configuration matching against the recovery, and a short local CPU integration check. That check used a reduced rollout of 64 total transitions and one update pass, verified finite weights and the gSDE distribution, saved/reloaded matching deterministic actions, and ran a four-decision control probe. The saved notebook retains the real 512-decision rollout setting and five update epochs. These checks confirm the basic connections work; they do not establish racing performance or Colab GPU throughput. I have not yet reported an actual long IMPALA run or a completed IMPALA lap.

### What I am doing before submission

My confirmed preserved recovery model is at 225,424 transitions. If GPU access and time allow, I will continue that same configuration toward a cumulative 500,000, keeping checkpoints and checking validation. That is a proposed experiment, not a result already achieved. Adding layers, changing exploration or increasing transitions has not been proved to repair the driving failure.

I need to compare saved candidates on the same evaluation tracks, using completion rate and DNF-penalized time first, then finishing lap times. A checkpoint named best can still fail every track. I should compare the selected candidate with the latest checkpoint instead of assuming the newest is fastest. A small initial IMPALA budget cannot fairly establish that its architecture is worse than a basic CNN trained much longer.

Because the deadline is close, I need to reserve time to export the required PyTorch weights and verify the organizer-compatible inference path, including image preprocessing, frame-history reset and valid actions. A training ZIP alone is not the finished submission interface. I have not yet recorded that the final inference adapter or official evaluation is verified.

What I can say honestly is that I have learned how to distinguish a failed driver, a display bug, a missing variable after restart and an unexplained GPU interruption. I have preserved weights and measured several failures. I have also seen partial road following, but I do not yet have evidence that this model is the fastest or reliably completes unseen laps. That is what the next evaluations need to establish.

## Where both runs are now, and starting the local config together

I reported that both training runs had continued. The basic CNN reached **337,710 transitions** and IMPALA reached **130,820**. Those are the last counters I supplied, not proof that either model has finished 500k. I have not supplied new race results for these counters or confirmed a backup file for each one yet. The earlier confirmed basic-CNN backup was at 225,424, even though its filename said 100000.

I want to keep both Colab runs going while implementing the local code here. I clarified that I mean the code matching the notebooks, not public-track fine-tuning yet. We agreed to work on **one code file at a time**, explain it in detail, then move through PPO, the basic CNN and the IMPALA CNN. For now we implemented only `src/ppo/config.py`. The other empty handwritten PPO and CNN files have not suddenly become working training code.

### What I understood about CPU, GPU and the time it takes

I was worried that continuing on CPU would lose the model's learning. It does not inherently erase the weights or make PPO learn a worse objective. A proper checkpoint resume keeps the saved weights, optimizer and counter. CPU and GPU can still produce different numerical trajectories; identical settings do not promise identical results.

The GPU handles neural-network computation when the model is actually on CUDA. CarRacing physics and image preparation still use CPU, and our DummyVecEnv runs its environments sequentially. This can limit how much faster the whole job becomes with a GPU. Similar reported progress over one hour is a rough observation, not a controlled benchmark: evaluations and the architectures also differ.

The counters alone do not prove which notebook uses a GPU. I need to check `torch.cuda.is_available()` and `next(model.policy.parameters()).device` in each separate runtime. Changing the text `DEVICE` does not move a model that already exists. A check cell cannot execute while the training cell occupies that runtime; it waits until training ends or I interrupt it.

When I need to pause, I click stop once, wait for the training cell's `finally` save attempt, and save a separately named backup before restarting or changing runtime. A hard runtime termination may skip that final save, which is why periodic checkpoints matter. Free Colab does not promise a fixed daily GPU allowance or reset time. Its resource warning is different from a CUDA traceback, and does not identify the device by itself. Colab's published rules also prohibit switching accounts to bypass resource restrictions. I have not established that both notebooks currently have GPU access.

At a hypothetical measured rate of 100,000 transitions per hour, the remaining basic-CNN budget from 337,710 to 500k would take around 1 hour 37 minutes, while IMPALA from 130,820 would take around 3 hours 42 minutes. Those are estimates at that assumed rate, not completion promises. If those hour counts already include evaluation time, I should not add the same overhead twice. Actual speed and runtime availability can change.

### The public-seed decision changed after clarification

I said that the organizer allows training on the three public seeds. This is my reported clarification; we have not independently seen an organizer message here. The document itself put public-seed exclusion in beginner advice rather than an explicit public-training disqualification rule.

Given that clarification, my proposed later experiment is to preserve a selected model and fine-tune a copy with mixed episode sampling: an initial idea is 80% ordinary training-pool resets and 20% public-track resets. That ratio is a starting experiment, not an established best value. Simply appending three seeds to the existing 100 would sample them only about 3% of the time, not 20%.

I still want held-out validation roads to check whether public improvements come at the cost of general driving. The final test stays untouched until selection. Public-track times would no longer be held-out generalization evidence after training on those seeds. Mixed training cannot guarantee no overfitting or the fastest mystery-track result. None of that fine-tuning is implemented or running as part of today's config file; current notebook settings still use the original random training pool.

### What the config file contains

`PPOConfig` stores the settings that define the current learning experiment. Its defaults match the basic-CNN gSDE recovery notebook. Choosing `architecture="impala_cnn"` matches the updated IMPALA configuration, with the encoder as the intended difference. These are settings, not code that runs the encoder or trains PPO.

The image settings are grayscale size 84, four stacked frames and action repeat 2. Two environments each collect 512 decisions, making a rollout of 1,024 transitions. Minibatches contain 128, with up to five passes per rollout. The learning rate is 0.0003, discount is 0.99, GAE lambda is 0.95 and probability-ratio clip range is 0.2. The critic weight and gradient-norm limit are both 0.5, but they do different jobs. The target KL is 0.03.

The exploration settings remain gSDE enabled, noise resampling every four decisions, log noise-scale initialization -2 and no extra entropy bonus. The CNN feature count is 256. The file also declares the separate seed groups; validation, test and public seeds are excluded from this baseline config. It does not enable the later public-track experiment.

`RunConfig` stores execution choices separately: target 500,000 total transitions, race limit 3,000 raw steps, saving every 25,000 transitions, evaluation every 100,000, device choice, progress display and output folder. This separation helps me understand why increasing a total target or switching compute hardware does not mean changing the model architecture.

I learned that `@dataclass` gives the settings a named container, and `frozen=True` helps prevent accidental reassignment during training. `__post_init__` checks common mistakes after creating the settings. The `rollout_size` property calculates `n_envs * n_steps`. `to_dict()` converts model settings to the same plain configuration dictionary the notebooks save; seed tuples become lists for JSON compatibility.

Checks confirmed that both architecture configurations match their notebook dictionaries and that the seed groups match the existing environment file. The default rollout calculation, frozen settings and some invalid-input guards were checked too. These are configuration checks, not a new local training run. Importing config starts no simulator, allocates no GPU and changes neither Colab run. We will explain this file before implementing the next code file.

## My basic CNN finally completed the validation lap

I reported that the basic-CNN run had completed its 500k target and shared `nice.mp4`. The evaluation output for validation seed **10000** was:

```text
finished: True
raw_steps: 1818
lap_seconds: 36.36
penalized_seconds: 36.36
reward: 818.199980661273
```

This is my first reported successful lap for the recovery model on this track. Earlier attempts on seed 10000 had either timed out while circling or ended after 190 raw steps near the playfield boundary. Those were failures, so their elapsed seconds cannot be treated as faster lap times. The new result finishes and receives 36.36 seconds instead of the 70-second DNF score.

The timing is consistent with `1818 / 50 = 36.36` simulator seconds. The video file is about 36.32 seconds long, which is consistent with the recorder skipping the terminal frame and recording at 25 frames per second. The browser message about video-tag support is a playback message, not evidence of a failed lap; the local file was readable.

The inspected frames show the car following several road sections and turns rather than spending the episode circling on grass. It sometimes runs near the road edge, so I should not describe it as a clean optimal racing line. Frames alone do not quantify throttle, braking, speed or how much time a particular corner loses. Those would need action logging or a more detailed rollout measurement.

The reward is consistent with the original full tile reward minus time cost: `1000 - 0.1 * 1818 = 818.2`, matching the reported value up to floating-point differences. This consistency supports the result, but the actual completion field and lap time remain the important evaluation evidence.

I should still print both `model.num_timesteps` and `evaluation_model.num_timesteps`. Completing a 500k training run does not prove the video used the final weights: section 14 may load an earlier `best_validation.zip`. The exact evaluated checkpoint counter has not yet been supplied for this video. I should preserve both the current model and the successful evaluated candidate before further training.

This is a clear improvement on one held-out seed, not proof that the model is the fastest, completes all validation tracks, or will finish the mystery track. It also does not isolate gSDE as the cause because training budget and policy weights changed too. The next comparison is the same ten validation seeds, 10000 through 10009, for the basic CNN and IMPALA at recorded budgets. I will report completion rate, average DNF-penalized time and finishing lap times, keeping final-test tracks untouched until selection. Any public-track fine-tuning remains a separate, preserved-copy experiment after that comparison.

## The successful video was the 400k model, not the latest weights

After I shared the current notebook, I confirmed `model.num_timesteps` was **500,526**, while `evaluation_model.num_timesteps` was **400,000**. This explains why the successful 36.36-second video and the later erratic control samples appeared contradictory. The video cell uses `evaluation_model`; the probe cell calls `model.predict`. They inspected different checkpoints.

The uploaded notebook contains stronger comparison evidence than the one video. Its periodic evaluation used the same first five validation tracks:

- At **400,000**, completion rate was **0.8**, or four finishes out of five. Mean DNF-penalized time was **38.504 seconds**, mean finishing time was **30.63 seconds**, and mean reward was about **813.7064**.
- At **500,000**, completion rate was **0.2**, or one finish out of five. Mean DNF-penalized time was **62.224 seconds**, mean finishing time was **31.12 seconds**, and mean reward was about **695.4123**.

The final current model is at 500,526, so the 500,000 callback measurement is not an exact evaluation of the final weights after the last update. Both should be evaluated explicitly before claiming they behave identically. Still, these logged results show a substantial deterioration between the two periodic checkpoints on the same five roads. Training reward or explained variance being positive cannot override that driving evidence.

`best_validation.zip` correctly remained at 400,000. That selection was intentional, not a failure to load the latest weights. The final 500,526-transition probe showed many steering commands at -1 or +1 and often little gas, but those samples do not describe the successful 400k driver. Nor does the deterioration alone prove overfitting: PPO learning instability and other causes have not been isolated.

The next action is to pause further basic-CNN training, preserve both candidates and compare them on seeds 10000 through 10009 without learning. I should use the selected candidate consistently for videos, control probes and export, rather than switching between `model` and `evaluation_model` unknowingly. The first five validation tracks already influenced checkpoint selection, so this ten-track comparison is model selection, not an unbiased final test. IMPALA should use the same ten tracks when ready. The final test remains untouched.

## IMPALA reached its reported 500k target, but the video still circled

I reported 500,000 total transitions for the IMPALA run and shared `huh.mp4` with about three hours left before my deadline. The inspected video lasts about 60 seconds. It shows the car leaving the road within the first few seconds, then spending much of the attempt circling and drifting on grass. This is a failed driving attempt, not evidence that the residual encoder automatically helped.

The current-model counter alone does not identify the weights used by the video. I still need the IMPALA `evaluation_model.num_timesteps`, actual gSDE settings and episode-result dictionary, because the basic-CNN video had previously used an earlier selected checkpoint. No multi-track IMPALA completion summary has been supplied for this stage. I should check its selected checkpoint before discarding the whole run.

A residual encoder provides a different image representation, not correct steering rules. Poor experience, policy optimization and settings that do not suit the encoder are possible contributors; the video does not isolate one learning cause or prove an implementation bug. A single run cannot establish that the IMPALA architecture is generally worse. Under my remaining time budget, however, the successful basic-CNN 400k checkpoint is the practical submission fallback, whereas extending a failed IMPALA experiment has an uncertain return.

The immediate priority is to preserve the successful basic candidate, finish comparable evaluation and build/verify the submission inference path. Public-seed training is allowed according to my clarification, but a rushed fine-tune could destroy useful behavior in its copy and consume the time needed to submit. It should only be an optional separate-copy experiment after a usable fallback and its loader are ready, with before/after public and held-out checks. Permission to train on the public roads is not evidence that it will improve them or preserve mystery-track performance.

## Ten-track comparison: I selected the 400k basic CNN

I supplied the full comparison on validation seeds 10000 through 10009. The selected 400,000-transition checkpoint finished **6 out of 10**, with mean DNF-penalized time **52.948 seconds**, mean finishing time **41.58 seconds**, and mean reward **751.4615**. The latest 500,526-transition checkpoint finished **4 out of 10**, with mean penalized time **61.758 seconds**, mean finishing time **49.395 seconds**, and mean reward **705.7742**.

The 400k candidate improves mean penalized time by **8.81 seconds** in this comparison and completes two more tracks. It is my preferred candidate from these two. The averages of finishing times use different successful-track sets, so the penalized averages are the clearer overall comparison. This is model-selection evidence on ten validation tracks, not an untouched final test or a guarantee about the mystery track.

For the 400k candidate, seeds 10000, 10001, 10002, 10004, 10008 and 10009 finished in 23.48, 43.62, 37.42, 57.10, 41.46 and 46.40 seconds. Seeds 10003, 10005, 10006 and 10007 timed out at 3,000 raw steps. The latest candidate finished seeds 10000, 10002, 10003 and 10004 in 57.78, 57.50, 39.78 and 42.52 seconds, and timed out on the other six.

The failed runs were not necessarily doing nothing. For example, the selected candidate's timeout reward on seed 10003 was about 688.42. With our unchanged reward, adding back the 300-point time cost gives about 988.42 out of 1,000 available first-visit tile reward. That is consistent with visiting most road tiles but not meeting the simulator's completion condition before the limit. The installed Gymnasium code terminates successfully when all tiles are visited or its new-lap flag is set; enough coverage alone does not guarantee the flag. This is a reason to inspect near-finish behavior, not to change the evaluator or call a DNF a finish.

There is a reproducibility detail I still need to resolve. The previous video labeled 400k on seed 10000 finished in 36.36 seconds, while this comparison reports 23.48 seconds. I have not established the exact saved-weight identity and runtime details across those runs, so I cannot call that difference a further training improvement. I should preserve a fixed selected checkpoint, log its identity and configuration, and repeat evaluation in a fresh matched inference path. The discrepancy does not erase either reported result, but means one seed and a timestep label alone are not enough to prove reproducibility.

The immediate next actions are to preserve the 400k candidate, evaluate it on the three public tracks under the clarified rules, and finish the submission loader/export checks. The 500,526 weights remain a backup. Public fine-tuning remains optional work on a separate copy, after preserving the selected model and its evaluation results. We have not yet supplied public-track results, final-test results or a verified official submission interface.

## My final decision about IMPALA and what I understood about training in batches

I decided to stop spending my remaining training time on IMPALA and set it aside for this submission. I watched it leave the road, drift and keep circling after a run that I reported had reached 500,000 transitions. With only about three hours remaining, I did not want to spend the deadline trying another long recovery. I want to preserve the experiment and its checkpoints as evidence, not delete it or pretend I never tried it.

My first explanation was: maybe I caused this by setting the target directly to 500k instead of training in batches. I thought smaller batches would be the solution. After separating the counters, I understand why that is not established by what we observed.

### A 500k target was already trained using batches

`TARGET_TOTAL_TIMESTEPS=500_000` does not tell PPO to drive for 500,000 transitions and perform one huge update at the end. It tells the training loop how long to keep collecting experience and updating the policy.

With our settings, two environments each collect 512 decisions. That makes a rollout of 1,024 transitions. PPO then estimates advantages and return targets, splits that rollout into minibatches of 128, and makes up to five shuffled update passes. Each pass contains eight minibatches when all are used. The approximate-KL safeguard can stop these passes early. Then PPO collects another rollout and repeats. An episode can continue across several rollouts; one rollout can also include an episode boundary.

So my 500k run already involved many smaller collections and updates. I did not accidentally disable minibatch training. A training target, rollout size, minibatch size, optimization epoch and race length are five different things.

Stopping at 100k, 200k and later targets would have let me watch earlier candidates, see deterioration sooner and choose whether to continue. That is useful experiment management. It does not automatically change the core PPO updates or guarantee a better driver. Changing `batch_size` or `n_steps` would change the learning setup, but we have not tested that as the cause or fix for IMPALA's drifting. Interrupting and resuming can also affect the exact sequence of experience, so I should not promise identical trajectories between every chunking strategy.

I will record my small-batch idea as a hypothesis I considered, not as the bug we proved or the solution we verified. The evidence supports setting IMPALA aside under this deadline; it does not support saying the IMPALA architecture cannot learn this game.

### What I can actually conclude from both models

The basic CNN had unsuccessful early policies too. The original ordinary-Gaussian 50,176-transition checkpoint circled and failed all eight diagnostic attempts. Its broad standard deviations and steering bias were measured, but we did not isolate a single training cause. The recovery changed a package of exploration settings, not just one parameter: gSDE, four-decision noise resampling, log-scale initialization -2 and zero additional entropy bonus.

The basic-CNN recovery later produced a successful driver. The important result was not that I reached 500k. It was that a preserved **400k checkpoint** finished six of ten validation tracks with mean penalized time **52.948 seconds**. The final **500,526-transition** checkpoint finished four of ten with mean penalized time **61.758 seconds**. That is an **8.81-second** advantage per track for the earlier candidate in this comparison.

The five-track periodic results had already favored 400k: four finishes versus one at the 500k callback. The callback correctly kept the earlier checkpoint in `best_validation.zip`, while `latest.zip` retained the later weights. Those files have different purposes. Selecting the best model is not the same as selecting the largest counter.

I also mixed up two policy objects while debugging: my successful video used `evaluation_model` at 400k, but my control probe used `model` at 500,526. The saturated steering samples from the latest model did not describe the successful candidate. I need to identify the exact checkpoint before explaining an action log or exporting weights.

I still have a reproducibility question: seed 10000 was reported as 36.36 seconds in the earlier 400k video and 23.48 seconds in the later comparison. Without pinned weight identity and matched runtime/inference checks, I cannot say why. A repeated timestep label alone is not enough to prove identical execution.

I have not supplied a comparable ten-track IMPALA report, and the timestep count of the weights used by its video has not been confirmed. That limits the scientific encoder comparison. My submission choice is a practical choice based on the successful basic-CNN evidence and remaining time, not a general architecture ranking from one video.

### The bugs, mistakes and fixes I should be able to explain

The all-DNF checkpoint-selection problem was a real selection bug: equal 70-second scores could leave the first candidate selected forever. The updated selector keeps mean penalized time primary and breaks ties with time-adjusted reward, retaining the off-map penalty. Focused selection checks passed. It does not change the training reward or teach the actor to drive.

The failure at exactly 100,000 transitions was a notebook display error. The traceback showed Rich and IPython recursively handling the validation-summary print. The training cell's finalizer reported saving the model. We disabled the Rich progress bar with `progress_bar=False` in the local notebooks and kept ordinary text logs. This avoids the implicated display path; the initial deprecation warning itself was not a failed PPO update.

The Python-version checkpoint-loading problem was a separate portability issue. The local viewer reconstructs the reviewed architecture and loads tensors rather than relying on a notebook class unpickled across Python minor versions. Short loading and action-match checks passed. It is not the cause of the original Colab circling.

The `NameError` after I restarted Colab meant the runtime no longer had `model` in memory. I needed to rerun definitions and resume from Drive, not start again or conclude that all saved learning was lost. The later GPU interruption still has no supplied exact traceback, so its cause remains unknown.

My backup named `backup_100000.zip` actually contained 225,424 transitions. The label I typed did not reset or change the weights. Using the current counter in future backup names avoids this confusion. Saving a notebook alone is also different from saving weights and optimizer state.

I learned not to call every bad behavior a code bug. Driving in circles, failing near the finish, an output exception, an exhausted resource allowance and a missing Python variable need different investigations. I also learned not to call a larger reward a faster race when the failed attempt ended earlier. A DNF still counts as 70 seconds.

### What I understand about the networks now

The CNN transforms the image history into learned features; it does not output a correct driving label. The actor uses those features to define the action distribution, while the critic estimates future return. PPO trains these paths using experience, advantages, action likelihood ratios and value targets. Both the actor and weighted critic loss can update the shared CNN. The 0.5 critic-loss coefficient and 0.5 gradient-norm limit are separate settings with different meanings.

For continuous actions, a Gaussian policy uses means and spreads to describe possible controls. A standard deviation is not an accuracy score. With gSDE, noise-weight scales combine with actor features, so a logged value near 0.13 is not one fixed action standard deviation for every image. No extra entropy bonus does not mean there is no exploration. Deterministic videos use mean controls and do not add this training exploration noise.

Residual connections in the IMPALA-style CNN add `x + F(x)` to help information and gradients pass through a deeper encoder. They do not encode the road rules or guarantee good driving. We used this encoder with PPO, not the distributed IMPALA learning algorithm. The updated notebook matches the recovery exploration setup, but it still needs measured driving evidence to justify using it.

### What I will preserve and finish before submission

I will use the selected basic-CNN candidate as my fallback and stop extending IMPALA for this deadline. I will preserve both experiments, the ten-track comparison and the failed behaviors in my record. I will not claim the 400k model is the fastest possible driver or guaranteed to finish the mystery track.

I reported that the organizer allows training on the three public seeds. That supersedes treating the beginner tip as an explicit ban. A later mixed public/random-track fine-tune would be a separate-copy experiment with before/after checks; it is not already implemented, trained or proved successful. Public-seed performance after training on them would no longer be held-out evidence. The final-test tracks remain separate.

With the time left, the priority is to preserve the chosen checkpoint, measure its public-track results, export the exact selected weights and verify a working inference interface. The exported `.pth` bundle is a dictionary of tensors and metadata, not a runnable model just because its suffix is `.pth`. The organizer's example leaves the inference step as a placeholder, so loading, preprocessing, frame-history reset, action bounds and repeat timing still need verification. I have not recorded a completed official submission or a verified final adapter.

For the local implementation, we intentionally stopped after `src/ppo/config.py` so I could understand one file before moving through PPO and then the CNNs. Its default learning settings were checked against the two gSDE notebooks. I should not say the empty local PPO modules have been implemented just because the notebooks already train through SB3.

My main lesson is to judge checkpoints by measured driving and to keep explanations tied to evidence. Training longer gave me a useful earlier policy and then a worse later policy. Watching, preserving and evaluating checkpoints mattered more than reaching a large number or assuming that a deeper CNN must win.

## Implementing the full local workflow and setting IMPALA aside

I asked to implement the whole local workflow, explain every file and function, commit the changes, and make the README describe the application professionally rather than present a learning-phase plan. This extends our earlier one-file-at-a-time work: config is no longer the only implemented local component.

I also reported that IMPALA's selected best checkpoint was at **100k**, even though its training run had reached about 500k. The screenshot still showed drifting/circling on grass in the displayed validation attempt. The 100k label means it was the best candidate under that run's selection history, not that it was a successful driver. I have not supplied its complete validation report or the selected weight file for local inspection, so those remain limitations.

I stopped IMPALA because of the deadline and its poor observed driving, not because we proved the architecture cannot work. Hyperparameter tuning, checking the full checkpoint-selection history, testing another training seed, inspecting exploration/action saturation, or increasing the budget after verifying learning are possible later experiments. A lower learning rate is an example to test, not a known solution. The run already performed repeated rollouts and minibatch updates; changing the total target from 500k into shorter chunks would mainly let me monitor it sooner, not automatically fix the policy.

### What is implemented locally now

Both image encoders now have real implementations under their existing folders. The basic CNN has three convolutions and a projection to 256 features. The IMPALA-style encoder has three pooling stages, six residual blocks and the same feature-output size. Layer names are compatible with the notebook weights. Neither encoder itself chooses a road seed or supplies a correct driving label.

The environment module now builds the complete notebook pipeline: a raw 3,000-step TimeLimit, seed sampling, action repetition, grayscale/resize, Monitor, vector transpose and four-frame history. It imports seed groups from config so those definitions stay aligned. The sampler is uniform over the ordinary training pool; public-track fine-tuning is still not enabled.

The policy module chooses the encoder and builds shared CNN features with separate 64-unit actor and critic heads. SB3 still supplies its ActorCriticCnnPolicy, Gaussian/gSDE distributions and optimizer construction. This is not a claim that we wrote every library component ourselves. The local PPO model routes collection and optimization into the project's own modules while retaining SB3's outer learn/log/save infrastructure.

`rollout.py` makes the collection steps explicit. It runs policy inference without gradients, sends bounded controls to the game, stores the original Gaussian samples and old likelihoods, handles time-limit bootstrap from the terminal image, and calls local GAE. It does not bootstrap from the next race's reset image. Its buffer subclass uses SB3 storage and minibatch generation while delegating advantages and returns to our function.

`gae.py` loops backward through the rollout, calculating temporal-difference errors and discounted advantage accumulation. Episode boundaries stop the recursion. `loss.py` computes the clipped actor objective, critic mean squared error, entropy term, approximate KL and clip fraction. Old likelihoods and targets are detached. `update.py` loops through epochs and shuffled minibatches, checks the KL guard, backpropagates the combined loss, clips gradients and updates parameters. Both losses can reach the shared CNN. The local KL log averages all visited minibatches, which differs from SB3's final-epoch KL logging but does not change the update mathematics.

The trainer can start fresh or resume toward a cumulative target, prints the actual device, writes settings/runtime versions, and attempts an atomic save when interrupted. The callback saves every 25k by default and evaluates five validation tracks every 100k. It saves a new best candidate before printing the summary, so a display exception cannot interrupt that selection step in the same way as the earlier notebook issue.

### Loading, comparison and export are separate jobs

The checkpoint module reads native JSON metadata and tensor states from this project's SB3 ZIPs without unpickling embedded notebook classes or lambdas. For training resume it reconstructs the known policy and restores Adam state, transition counter and update counter. It deliberately resets simulator observations; it does not recreate partial rollouts, exact RNG continuation or old episode logging history. Its supported notebook recipe uses a constant learning rate.

The evaluator can compare multiple checkpoints on exactly the same validation seeds, print each race and save summaries with configuration, counter, policy-file fingerprint, device and package versions. It uses completion metadata rather than interpreting any terminated episode as a successful lap. Short raw-step limits are marked as debugging. Mean penalized time remains the primary comparison; finisher-only means can hide failures.

The export writes a CPU-tensor bundle with settings, training counter and source fingerprint. It remains a weights dictionary, not a pickled model object with an `.eval()` method. `load_driver` reconstructs the policy and supplies a `Driver` with `reset` and `act` methods. The driver accepts only raw `(96,96,3)` RGB uint8 images, keeps zero-padded history and repeats controls at the same timing as training. It needs a call after every raw simulator step, and a reset between episodes. Adding another repeat wrapper around that raw-step interface would repeat actions twice incorrectly.

The export verifier compares the raw-step driver with the original vector policy in separate seeded games. It checks deterministic actions, held actions, reward sums and ending flags. I still need to connect this interface to the organizer's actual callable contract: the provided evaluation reference leaves that part as a placeholder. A working local adapter is not the same as organizer acceptance.

### What was actually checked

Five numerical regression tests passed. They check local GAE against SB3 including reset and timeout cases; PPO clipped-loss values and detached old targets; local collection against SB3 on short truncating episodes; actual optimizer-update parity against SB3 rather than two identical early exits; checkpoint/Adam continuation; and DNF tie selection without early-crash bias. Some tests cover several of those behaviors in one method.

Both architectures also completed a short real-CarRacing integration run: 64 training transitions with smaller test rollout/batch settings, a 16-raw-step debug episode limit, periodic checkpoint saving and five-track validation callbacks. These runs test the connections, not racing quality. Their temporary debug checkpoints were kept outside the repository and do not replace the Colab candidates. A command-line two-checkpoint comparison also completed with explicitly shortened debug episodes.

The tensor export and inference comparison passed for the available local Colab file `latest(3).zip`, whose native counter is 500,526. It ran through **2,671 raw steps** on seed 10000 and reported a finish; the adapter and reference policy matched within that run. That local time would be 53.42 seconds. This is not a new training improvement, and it differs from the earlier Colab timing of that counter, so it reinforces the need to record exact checkpoint identity, package versions and execution details. The local machine's PyTorch/Python environment differs from Colab. I cannot assert a unique cause for the timing difference.

A separate short IMPALA export check passed for the temporary 64-transition smoke model over 32 raw steps. That is portability coverage, not a successful IMPALA race. The actual selected 400k basic-CNN ZIP is not currently available in the local Downloads files inspected, so its final export parity still must be run once that exact file is supplied. We must not accidentally submit the local 500,526 test export instead of the selected model.

### README, explanations and remaining submission work

The README now explains CarRacing, the two encoder architectures, local setup, fresh training, resume, comparison, videos, export and the raw-image inference interface. It identifies PPO with the basic CNN and gSDE as the current submission candidate. The IMPALA alternative is documented honestly as set aside, rather than removed from the project or presented as a proven superior model.

The separate implementation reference explains every file and its functions. SB3 attribution and its MIT notice are included because the local collector/update semantics follow its implementation. Source code, tests and documentation are committed; generated weights, videos, virtual environments and caches are excluded from Git. The previously excluded planning/research documents remain separate from this implementation commit.

What still needs real submission evidence is the exact selected 400k export, organizer-compatible entry-point invocation, public-track results and the final evaluation chosen after model selection. No additional competitive local training, successful public fine-tune, or official acceptance is claimed by implementing and testing these files.

### Withdrawing the public-seed training proposal

I asked to remove the proposed public/ordinary mixed-training notebook and its instructions. After checking the written guideline again, it says to use the public seeds only to test. I am no longer relying on my earlier uncertain permission statement to recommend training on them. The selected 400k baseline stays preserved. Any separate Colab experiment already run is not undone by deleting a local notebook, and its results must not be described as unseen-track results if it trained on those tracks.

My next question is how to reduce left/right steering oscillation while preserving completion and improving lap time. A friend reported improving stability, but I do not yet know which change helped their model. Possible future experiments should use ordinary training roads and compare on separate evaluation roads. No steering penalty, filter, new reward or improved model has been implemented or validated by this cleanup.

### Exporting my actual selected 400k weights

I supplied `selected_400000(1).zip` from Downloads. Its saved counter is exactly 400,000. We exported `submission/carracing_basic_cnn_400000_weights.pth` with the policy tensors and matching settings, and added its link, SHA-256 and loading instructions to the release notes. The PTH is an inference bundle; I keep the original ZIP for optimizer state and training continuation. Adding a link to Markdown does not upload the binary to a GitHub release.

The complete local export comparison passed on seed 10000. The exported driver and original checkpoint produced matching actions, repeated controls, reward sums and ending flags, finishing at 1,818 raw steps, or 36.36 seconds. This agrees with my earlier video and differs from the later reported 23.48-second run. We still cannot assume those two timings used identical weights and runtime conditions. This export check verifies conversion, not faster fine-tuning or performance on every road. No release has been published and the original ZIP was not changed.

### Publishing a baseline while fine-tuning continues

With about half an hour remaining, I chose to publish the verified 400k basic-CNN baseline rather than wait with no downloadable model. The release includes its inference weights, original training ZIP, checksums, reports and loading instructions. It clearly separates the reported ten-track Colab results from the local export-parity check and leaves the timing discrepancy visible. The separate fine-tuning run has not replaced this baseline and has no measured improvement claimed here. Publication is distribution of the candidate, not proof of official submission acceptance.

### Removing the discarded notebook

The public-seed mixed-training notebook and its run instructions have been removed at my request. The local trainer still rejects a checkpoint with a different sampler instead of silently changing its training distribution. General checkpoint metadata remains compatible with previous exports. My baseline weights and original training checkpoint are unchanged. Earlier journal entries remain historical discussion, not current instructions or confirmed organizer permission.

### A separate steering-stability experiment on ordinary training roads

I asked for a small experiment that could fit the remaining time. The new local notebook starts from a protected copy of the selected 400k model, uses the same CNN/PPO/gSDE structure and trains only on ordinary seeds 1000–1099. The discarded public-road sampling proposal remains removed. This new notebook has not been published or used to replace my released model.

The hypothesis is to subtract 0.01 times the squared change in executed steering once per policy decision. This may reduce abrupt left/right oscillation, but it might also discourage useful corrections. The first decision after an episode reset is unpenalized, and constant steering has no change cost. We keep the game's progress/time reward and add no gas bonus. Evaluation uses the original reward and no steering filter. These are trial choices, not a confirmed improvement.

The experiment also tests a learning rate of 0.0001. Since the reward and learning rate both change, a result would not isolate their individual effects. It caps training near 10k additional transitions or about 12 minutes per invocation, leaving time for three validation roads before and after. Individual updates and saves can overrun the timer; setup and evaluation are additional. A timer stop during rollout collection discards the incomplete rollout but keeps earlier updates. Transition count alone does not prove how many updates happened.

The notebook's syntax passed. A short test checked steering-cost arithmetic, episode reset, unchanged reward on evaluation environments, rejection of public seeds in training, independent parent weights, a real 32-transition continued PPO update at the reduced Adam rate, timer stopping and checkpoint metadata. This test used a temporary 64-transition fixture, not competitive training of my 400k model. No faster driving has been measured. I will retain the baseline unless the candidate finishes at least as reliably and improves penalized time. The ordinary local trainer refuses custom-reward checkpoints unless their matching reward wrapper is supplied through the appropriate workflow.
