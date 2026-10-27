# My current training plan

My goal is a driver that finishes quickly and reliably on roads it has not practiced. This plan replaces the earlier planning-only version. I have a measured failed baseline, and a separate recovery experiment has completed its first chunk at 25,600 transitions. I have its training log, but no recovery driving evaluation proving that the problem is fixed.

## What the competition requires, and what I choose

The [challenge guidelines](<carracing_ai_challenge_guidelines (1).md>) allow 3,000 raw simulator steps, or 60 simulator seconds, per race. A DNF is scored as 70 seconds. Each competition track is run three times. The guidelines recommend varied random training seeds and keeping the three public tracks for testing.

They do not require 25,000, 100,000 or any other specific number of training transitions. Those are my experiment budgets. The earlier 17.8 minutes was measured training time, not a competition requirement. My result must be judged by driving, not by reaching a counter.

## The pipeline I am training

The policy receives images, not the seed or hidden car coordinates:

```text
96 × 96 RGB observation
→ grayscale and resize to 84 × 84
→ stack four recent decision-time frames
→ CNN produces 256 learned features
→ actor head produces an action distribution
→ sampled controls during training / mean controls during deterministic evaluation
→ steering, gas and brake move the car
```

The critic uses the shared CNN features through a separate head to estimate future discounted reward. PPO is the algorithm that collects experience and updates these networks. It is not another network that receives the features. The image pipeline, frame-history reset and action handling must match between training and evaluation. SB3 normalizes the uint8 image once, not twice.

The original reward is unchanged: -0.1 per raw simulator step, a first-visit reward of `1000 / number_of_track_tiles` for each new road tile, and a -100 step reward on leaving the playfield. Revisiting the same tiles does not keep earning their reward. The 70-second DNF score is separate from the training reward.

## Keep track groups separate

- **Training:** seeds `1000–1099`, a starting pool of 100 tracks.
- **Validation:** seeds `10000–10019`, used for settings and checkpoint selection.
- **Final test:** seeds `20000–20019`, untouched until I choose the model and settings.
- **Public reporting:** seeds `202`, `42` and `203`, excluded from training.
- **Mystery competition track:** unknown until the organizer selects it.

The numbers identify roads; higher seeds do not mean harder roads. One hundred tracks is not a proven sufficient number. At each episode reset, a training worker samples from the pool uniformly with replacement. Tracks can repeat before all 100 are visited. I am training one shared driver, not a separate driver for each track.

The run seed controls initialization and sampling randomness. A track seed controls road generation. I will preserve settings and package versions, and record track exposure in future experiments before making claims about which training tracks were actually practiced. The original checkpoint's episode history did not record that exact coverage.

## Understand the counters before changing them

One transition is one observation/action/reward/next-observation interaction. It is not an episode. With action repeat 2, a decision advances at most two raw simulator steps, stopping early if the episode ends. The 3,000-step time limit remains inside action repetition.

With two training environments, both contribute to the total transition count. Each contributes 512 decisions per rollout, giving `512 × 2 = 1,024` transitions before an update phase. PPO splits that rollout into minibatches of 128 and can make up to five shuffled passes; its approximate-KL stopping condition can end updates earlier. An episode can continue across rollout boundaries, and a rollout can contain several episodes.

I will initially keep these rollout and minibatch settings unchanged. Shorter training chunks let me inspect progress more often; they are not smaller minibatches and do not shorten the race. PPO can slightly exceed a requested cumulative budget to finish a rollout.

## What failed in the measured baseline

The first basic-CNN checkpoint reached 50,176 total transitions. Its longer segment took 17.8 minutes at about 46 transitions per second. The saved episode history contained only 36 completed episodes. It could not have completed an attempt on all 100 tracks.

The deterministic video on validation seed `10000` showed circling on the grass, then a timeout at 3,000 raw steps with reward about -280.77. Inspection of the saved weights reproduced failure on two training-pool tracks and two validation tracks. Steering stayed positive throughout those deterministic attempts, with means around 0.65–0.67. Four additional sampled-action attempts also failed. There were zero finishes across eight full episodes.

The policy weights were finite, the inspected images were normalized once, and reconstructed predictions matched the Colab action samples. No obvious image/clipping/time-limit mismatch was found. This is not proof that every possible bug has been excluded. It establishes a poorly learned turning policy, not a unique explanation for why training learned it. Pool membership does not prove that a particular road had already been visited.

Broad exploration and limited useful experience are plausible contributors. They are not established as the only causes. The measured Gaussian standard deviations were about `[1.013, 0.980, 0.973]`. Turning evaluation randomness on did not solve the failure. Adding CNN layers is not an evidence-based repair at this stage.

The [full baseline measurements](../notebooks/results/baseline_50176_diagnostics.json) are preserved for comparison.

## What is fixed, and what is still an experiment

**Checkpoint selection:** all-DNF checkpoints tie at 70 seconds. The notebooks now break equal-time ties with higher time-adjusted raw reward, `reward + 0.1 × raw_steps`. This adds back the original time penalty while retaining any off-map penalty. It is a progress proxy, not an exact tile count. Raw reward alone could prefer an early crash simply because less time elapsed. Penalized time remains the primary score. This selection fix passed focused checks; it does not repair the actor's behavior.

**Local loading:** the viewer reconstructs the reviewed architecture and loads tensor weights instead of unpickling a notebook class saved under a different Python minor version. It matched the uploaded policy's predictions. That portability fix is separate from the car circling in Colab.

**Recovery:** the [gSDE recovery notebook](../notebooks/ppo_cnn_gsde_recovery.ipynb) keeps the basic CNN, image pipeline, original reward and track groups. It uses:

```python
use_sde = True
sde_sample_freq = 4
log_std_init = -2.0
ent_coef = 0.0
```

The [SB3 maintainers' CarRacing recipe](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) motivates these exploration choices. This is not their complete recipe; their network, preprocessing, normalization, optimization and budget also differ. Changing this package of exploration settings does not isolate one causal parameter.

In gSDE, sampled noise weights act on policy features and are resampled every four decisions. Features can still change the action between resamplings. The -2 setting is a log-scale initialization of noise weights, not a negative standard deviation or a fixed action standard deviation of `exp(-2)`. A zero entropy bonus does not remove action sampling.

The recovery's short local checks passed collection, an update, deterministic prediction, saving/reloading and a shortened evaluation. Its first Colab chunk reached 25,600 total transitions, with the longer segment collecting 24,576 in 8.6 minutes. The final displayed mean reward was about -245, approximate KL 0.0126 and explained variance 0.574; critic loss fluctuated substantially earlier. These are training diagnostics, not proof of successful driving. The detailed interpretation is in the [journal](journal.md#the-failures-we-found-the-verified-fixes-and-my-recovery-experiment). I will inspect a deterministic validation video before continuing the same model toward 50,000 and 100,000 transitions, and will not call the recovery a successful fix until held-out driving improves.

## Continue the recovery in small, resumable chunks

1. Preserve the original `basic_cnn/run_01/latest.zip` and its results. Use the separate `basic_cnn/gsde_recovery_01` folder for recovery.
2. If the current Colab training cell is running, let it finish. Do not restart or change settings during that segment.
3. For a fresh runtime, select an available NVIDIA GPU, run setup, mount Drive, and define the wrappers and networks. For a new run only, leave `RESUME_FROM = ""`. To resume an existing recovery run, set `RESUME_FROM = str(RUN_DIR / "latest.zip")` before loading the model. Do not load the old diagonal-Gaussian checkpoint into the gSDE policy.
4. Run the short pipeline check once as needed. On a resumed run it can be skipped with `RUN_SMOKE = False`. A passing smoke check is not a driving result.
5. Use cumulative section-12 targets of 25,000, then 50,000, then 100,000 transitions. Continue the same model. After initial setup, rerun only the necessary cells rather than Run all.
6. After a segment, confirm the counter and checkpoint, inspect logs, select the evaluation model in section 14, and record a validation video in section 15. Check which checkpoint and timestep count are actually being watched.
7. If progress and available compute justify it, extend toward 250,000, then further measured budgets. These are inspection points, not guaranteed convergence thresholds. If the car remains stuck, inspect controls and training-pool performance before blindly adding more time or layers.
8. At serious comparison points, enable full validation on all 20 validation tracks. Leave the final-test switch off until selection is finished.

The current callback checks five fixed validation tracks every 100,000 transitions. Before that first check, `best_validation.zip` may not exist; use `latest.zip` explicitly. "Best" means best among evaluated candidates, not necessarily a competent driver. Checkpoints and results go to Drive; saving the notebook alone does not preserve model state. Back up the session's TensorBoard logs after training segments.

## Decide from evidence

My main measure is average DNF-penalized time, supported by completion rate and finishing lap times. I will also inspect training reward, controls, videos, policy-change statistics and critic prediction quality. A lower critic loss does not establish better driving.

Poor training-pool and validation results motivate investigating basic learning. Good measured training performance with consistently poor validation motivates investigating generalization and increasing training diversity. One track cannot establish overfitting. Validation itself can be overfit through repeated tuning, which is why the final test stays separate.

Continuing the original baseline remains a useful control if compute allows. Comparisons need comparable budgets and recorded transitions and wall-clock time. If I later compare the IMPALA CNN, I must hold the exploration and evaluation setup fixed too. Comparing the original diagonal-Gaussian IMPALA notebook directly with the gSDE basic CNN would change more than the encoder. The residual encoder is a later experiment, not an automatic solution to circling.

## Watch, test and prepare the final artifact

In section 14, select the checkpoint and inspect full validation; in section 15, record its deterministic driving. Only after model selection will I enable the 20 final-test tracks and report the public tracks. Repeating a deterministic policy on an identical seeded environment may repeat the same run, so three repetitions are not three distinct generalization examples.

To watch a downloaded checkpoint locally:

```bash
cd /home/dahmane/dev/max-verstappen
.venv/bin/python src/watch_agent.py --model /home/dahmane/Downloads/best_validation.zip --seed 10000
```

Use the actual downloaded filename. To record instead of opening a window, add `--video /home/dahmane/Downloads/trained_car.mp4`. The viewer supports both original and recovery policies and does not train. `src/set_up.py` is the random-action demo, not the trained driver.

For submission I still need the required PyTorch `.pt`/`.pth` artifact with a verified inference adapter matching the organizer's loader, plus source and journal. The SB3 `.zip` is a resumable training checkpoint, and a raw `.pth` bundle alone is not a complete inference interface. I must verify preprocessing, reset, action handling and offline loading before treating the submission as ready.

My detailed explanation of the CNN, Gaussian sampling, PPO, GAE and gradient flow is in the [journal's consolidated learning entry](journal.md#what-i-understand-now-and-my-next-plan).
