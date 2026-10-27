# Local implementation reference

Run entry points from the repository root with `python -m src.ppo.<module>`. Importing the configuration or model modules does not start training. The Colab notebooks remain self-contained; editing local files does not change an already uploaded notebook.

## Execution flow

During training, the environment supplies images, the shared encoder creates features, the actor samples an action and the critic predicts a value. The collector stores fresh experience. GAE produces fixed advantages and return targets. PPO reevaluates the recorded actions, computes losses and updates weights through minibatches. Validation uses separate environments and never updates weights.

During inference, the driver receives one raw RGB image per simulator step. It maintains its own frame stack and action-repeat counter, calls the policy only at decision times, and returns the clipped deterministic mean action. Its reset method separates episodes.

## `src/ppo/config.py`

**`PPOConfig`** contains model and learning settings, not a running model. The architecture flag chooses basic or residual image features. `run_seed` controls initialization/sampling; the training seed tuple controls roads. The image settings, rollout settings, optimizer settings and exploration settings match the current gSDE notebooks.

- `PPOConfig.__post_init__` validates architecture names, positive dimensions, minibatch divisibility, finite coefficients, seed range and excluded evaluation groups. Evaluation groups are excluded from training.
- `rollout_size` returns `n_envs * n_steps`: 2 * 512 = 1024 by default.
- `to_dict` produces native JSON/checkpoint metadata, converting the training-seed tuple to a list.

**`RunConfig`** contains cumulative target, raw episode limit, save/evaluation intervals, device and destination folder. It is separate so increasing a budget does not appear to change PPO hyperparameters. `RunConfig.__post_init__` rejects invalid budgets, device names and experiment path components. A frozen dataclass prevents accidental field assignment; create another instance for a different experiment.

## `src/ppo/environment.py`

- `make_env` creates a raw RGB continuous-action game with a 3000-step TimeLimit. It does not reset, choose actions or train.
- `sample_training_seed` chooses one of the original training roads with replacement using a persistent NumPy generator.
- `reset_training_episode` resets that road and returns its seed as logging metadata, not policy input.
- `reset_evaluation_episode` validates an explicit validation/test/public seed before reset.
- `TrackSampler.__init__` stores a seed pool, ordering mode and independent sampling generator. `reset` optionally seeds the sampler and chooses the road; `step` attaches the current road identifier to metadata. The observation remains an image.
- `ActionRepeat.__init__` sets repetition and raw-step counter. `reset` clears the counter. `step` repeats a control, adds raw rewards and immediately stops repetition on termination or truncation. It reports actual raw steps.
- `GrayResize.__init__` declares the `(84,84,1)` uint8 output space. `observation` applies Pillow grayscale and bilinear resize. It does not normalize pixel intensity; SB3 performs float conversion and `/255` once.
- `build_vec_env` constructs each worker in an inner `factory/create` closure that binds its rank. The ordering is raw TimeLimit -> sampler -> repeat -> grayscale -> Monitor -> DummyVecEnv -> transpose -> frame stack. The TimeLimit must remain inside repetition. Four stacked frames have shape `(4,84,84)`, and missing reset history is zero-padded. DummyVecEnv batches neural-network observations but runs physics serially.

## `src/ppo/models/cnn/network.py`

- `BasicCNN.__init__` creates convolutions `4 -> 32 -> 64 -> 64`, using kernels/strides `8/4`, `4/2`, `3/1`. The spatial sizes become 20, 9 and 7. A zero tensor determines the flatten size; the projection maps 3136 values to 256 features with ReLU.
- `BasicCNN.forward` applies convolution and projection to already normalized observations. The output is a learned representation, not steering, a reward or 256 human-labeled road properties. Parameter names match the notebook for strict checkpoint loading.

## `src/ppo/models/impala_cnn/network.py`

- `ResidualBlock.__init__` builds two same-channel 3x3 convolutions with preactivation ReLUs. `forward` returns `x + change(x)`, keeping a shortcut path for information and gradients.
- `ImpalaStage.__init__` changes channels, downsamples with padded max pooling and adds two residual blocks. `forward` runs that stage.
- `ImpalaCNN.__init__` combines stages with 16, 32 and 32 channels, then flatten/projection to 256 features. Spatial sizes are 42, 21 and 11. There are six residual blocks, twelve block convolutions and three stage convolutions: fifteen convolutions in total.
- `ImpalaCNN.forward` applies these stages to normalized pixels. Residual connections do not guarantee better racing or remove the need for optimization and evaluation. This encoder is trained by PPO, not by IMPALA/V-trace.

## `src/ppo/policy.py`

- `policy_kwargs` chooses the encoder class and specifies shared features, 256 output features, separate 64-unit actor/critic hidden layers, normalization and initial log noise scale.
- `make_policy` creates observation/action spaces and SB3's `ActorCriticCnnPolicy` without a simulator. It is the inference reconstruction path. Orthogonal initialization is skipped because loading replaces every learned parameter. The actor outputs three Gaussian means; the critic outputs a scalar value. In gSDE, a learned 64-by-3 noise-scale matrix controls feature-dependent exploration, not one constant action spread per image.

SB3 supplies the continuous action distribution, mean/log-density calculations, Adam optimizer construction and policy tensor preprocessing. This project does not claim those are implemented from scratch. Deterministic `predict` uses the mean and legal action bounds. For stochastic gSDE inference, reset noise explicitly; training's automatic resampling does not occur merely by calling `predict(deterministic=False)`.

## `src/ppo/rollout.py`

- `GAERolloutBuffer` subclasses SB3 storage and minibatch generation. `compute_returns_and_advantage` converts final critic predictions to NumPy and calls local `compute_gae`.
- `collect_rollout` clears the buffer, puts the policy into evaluation mode for collection, and samples gSDE noise at rollout start and the configured frequency. It predicts actions/values/log densities under `no_grad`, executes bounded controls and increments the counter by the number of workers. Callback and Monitor information are updated at each vector step.

Crucially, it stores the original unbounded Gaussian sample with its original likelihood, while the game receives a clipped copy. It does not pair a clipped sample with the wrong original density. At a TimeLimit truncation, it adds `gamma * V(terminal_observation)` to the stored reward. The final image comes from `info`, because the vector environment's returned observation has already reset. True terminations get no future bootstrap. At the end of an unfinished rollout it predicts the next value, computes GAE and signals the callback. Returning False means the callback requested stopping before a full rollout.

## `src/ppo/gae.py`

- `compute_gae` accepts arrays shaped `[rollout_steps, environments]`. It validates dimensions and loops backward. The temporal-difference residual is `reward + gamma * next_value * continues - value`; the recursive advantage adds `gamma * lambda * continues * next_advantage`.

An episode-start mask at the next step stops recursion between races. At the last collected step, the `dones` mask determines whether to use the supplied final value. Timeout rewards have already received their terminal-image bootstrap in collection; their recursion still stops, preventing the next race from leaking into this one. Returns equal advantages plus old values. All are fixed data for optimization.

## `src/ppo/loss.py`

- `LossTerms` groups the combined loss, actor loss, critic loss, entropy loss, approximate KL and clip fraction so logs keep their distinct meanings.
- `ppo_loss` flattens minibatch vectors, detaches old log densities/advantages/return targets and optionally normalizes advantages. It computes `ratio = exp(new_log_prob - old_log_prob)` for the same recorded action. The actor minimizes the negative mean of the smaller unclipped/clipped advantage objective. The critic uses mean squared error against the fixed return target; optional value clipping is supported for compatible callers. The entropy loss is negative entropy, or a log-density estimate when analytic entropy is unavailable.

The total is `actor + vf_coef * critic + ent_coef * entropy_loss`. With current settings, the entropy coefficient is zero, but sampling and learning exploration scales still occur. Both actor and weighted critic gradients reach the shared CNN. The critic head receives the critic gradient; the mean head receives actor-objective gradients. Standard deviations are learned distribution parameters, not confidence or accuracy percentages. Continuous likelihoods are densities, not probabilities of exact real-valued actions.

Approximate reverse KL is the mean of `exp(log_ratio) - 1 - log_ratio`; clip fraction counts ratios beyond the clipping interval. These metrics are computed without gradient tracking. Neither is a crash rate.

## `src/ppo/update.py`

- `update_policy` activates training mode, refreshes the learning rate and obtains the current clip range. For each epoch it asks the buffer for shuffled minibatches, reevaluates recorded actions and calls `ppo_loss`. It rejects a non-finite total. If approximate KL exceeds `1.5 * target_kl`, it skips that step and stops the current update phase, not the whole training job.

For accepted updates, `zero_grad` clears previous gradients, `backward` propagates the combined objective through both heads and the shared encoder, gradient clipping limits the combined norm, and Adam changes the weights. Losses are calculated values, not stored parameters that are themselves updated. The default 0.5 gradient bound and 0.5 critic coefficient have different purposes. Logs average minibatch diagnostics across this phase; the logged final total is the last minibatch loss. The local KL log aggregates all visited minibatches, whereas SB3's implementation logs its final epoch's KL average. This logging difference does not change the gradients.

## `src/ppo/model.py`

- `LocalPPO.collect_rollouts` routes SB3's outer training loop to local collection.
- `LocalPPO.train` routes each update phase to `update_policy`.
- `make_model` builds a fresh LocalPPO, applies every typed configuration field, chooses our GAE buffer and attaches native experiment metadata. Initialization and resuming are separate operations.

The outer SB3 `learn` loop still handles cumulative budgeting and progress accounting. This local trainer is intentionally continuous Box-action CarRacing PPO, not a generic replacement for all SB3 spaces and algorithms.

## `src/ppo/checkpoint.py`

- `read_checkpoint` recognizes the SB3 ZIP layout and reads native JSON plus `policy.pth` with `weights_only=True`. Alternatively it reads our exported tensor bundle or the notebook's plain weights/config bundle. It does not deserialize embedded notebook classes or lambda schedules.
- `config_from_metadata` normalizes missing legacy Gaussian settings, restores seed tuples and validates the known shared/head layout. Unknown configurations are rejected rather than guessed.
- `load_policy` constructs the known inference policy, strictly loads tensors, checks finiteness and returns the policy/config/counter. No optimizer is needed for inference.
- `resume_model` creates LocalPPO with new environments, strictly loads policy and Adam states, restores the transition/update counters and leaves observations unset so `learn` resets the new simulator. This is the supported constant-rate notebook recipe. It does not restore a partial rollout, exact RNG state or old episode logging history.
- `save_checkpoint` writes to a temporary ZIP in the destination directory and atomically replaces the final path after a complete write. Naming a file `backup_100000.zip` does not change the internal counter.

## `src/ppo/callbacks.py`

- `RacingCallback.__init__` reads measurement history and initializes selection from the actual saved best-score record, not merely from an evaluation that might have failed to save.
- `_on_training_start` calculates the next saving/validation boundary from the resumed counter.
- `_on_step` periodically saves latest weights, evaluates five fixed validation tracks in separate environments, saves an improved best candidate, persists results and records validation metrics. It saves before printing so a display problem cannot prevent checkpoint selection. The tie-breaker uses time-adjusted raw reward only when primary race scores are equal.

Five-track selection is inexpensive but limited. The ten/twenty-track comparison is broader validation and still not an untouched final test. A model called best can fail every track.

## `src/ppo/train.py`

- `train` sets headless rendering and thread limits, validates CUDA availability and existing-run settings, writes configuration/runtime versions and builds environments. It creates or resumes a model, prints the actual parameter device and learns `max(0, target - current_counter)` additional transitions. It attempts an atomic save in `finally`, including on normal interruption, and always closes environments. Hard process termination cannot guarantee the finalizer executes.
- `main` parses architecture, budget, device, experiment paths and fresh-run hyperparameter overrides. Resume reads the checkpoint configuration and rejects incompatible overrides. It avoids silently overwriting an existing run. All CLI behavior is under a main guard; imports never launch training.

## `src/ppo/evaluate.py`

- `evaluate_tracks` validates split membership, creates a single seeded vector environment for each track, executes deterministic actions to termination/truncation and reads final completion metadata. Optional video encoding is streamed. It closes environment/writer even on an exception. Debug step limits are labeled, not presented as normal challenge scores.
- `summarize` reports completion rate, mean DNF-penalized time, mean finishing-only time and reward. It rejects empty input. Finishing-only averages exclude failures and must not replace the primary score.
- `selection_key` sorts primarily by mean penalized time. For ties, it prefers higher mean `reward + 0.1 * raw_steps`, which adds back time cost while retaining off-map penalties. It is not exact tile metadata and assumes unchanged rewards.
- `checkpoint_fingerprint` hashes the policy tensor file in an SB3 ZIP, or the exported file otherwise, to help distinguish candidates with identical transition labels.
- `main` evaluates one or several checkpoints on identical split seeds, prints each race, writes reports and prints the candidate selected from that comparison. It never trains or overwrites source checkpoints.

## `src/ppo/driver.py`

- `Driver.__init__` stores the loaded policy/config/counter and initializes episode state.
- `reset` fills the history with zero images and clears the raw-step counter and held action. Call after every environment reset.
- `eval` disables training behavior and returns the driver, mirroring PyTorch's familiar convention.
- `act` validates raw `(96,96,3)` uint8 input. On decision steps it applies the exact Pillow preprocessing, appends a frame and obtains a deterministic bounded action. On intermediate raw steps it returns the held action without adding an extra decision-time image. It returns a copy so the caller cannot accidentally mutate the held control.
- `forward` delegates to `act`, so PyTorch module calls `driver(observation)` use the same interface. The Driver subclasses `nn.Module`.
- `load_driver` loads an inference policy from ZIP or exported weights and wraps it in episode state. It needs no internet or game object and receives no hidden coordinates.

## `src/ppo/export.py`

- `export_weights` reads selected native metadata/tensors, writes CPU tensors with a format version, configuration, counter, package versions and source fingerprint, then strictly reloads it. It does not overwrite the source. The resulting dictionary needs `load_driver`; it is not a standalone pickled `nn.Module`.
- `export_driver` optionally saves an importable full Driver module for the organizer-style `torch.load(..., weights_only=False).eval()` interface. Load only the project's own trusted full-object artifact and keep source classes importable. The default tensor bundle remains available.
- `verify_export` runs a raw-step Driver and the reference vector policy in separate identically seeded simulators, comparing actions, repeated actions, accumulated rewards and ending flags. This checks preprocessing/history/repeat behavior across a rollout. Any mismatch raises an error; passing one seed is not a proof of all-track performance.
- `main` exports and optionally invokes parity verification. The default verification limit is the full 3000 raw steps; a shorter check is explicitly reported. The organizer's final callable interface is still external to this repository and must be connected to `load_driver/reset/act`.

## Existing viewer and tests

`src/watch_agent.py` remains the standalone desktop/MP4 ZIP viewer. Its `load_policy` reconstructs the notebook architecture and loads tensors; `watch` runs one deterministic race; `main` parses display/video arguments. It keeps equivalent encoder and wrapper definitions so its original direct-script command remains usable. Prefer the modular evaluator for new comparisons and the modular Driver for exported bundles.

`tests/test_ppo.py` checks GAE numerically against SB3, reset/timeout boundaries, clipped-loss values and detached targets, actual optimizer steps against SB3, policy/Adam continuation and DNF tie selection. Its synthetic short episodes exercise automatic reset without needing a competitive driver. Real CarRacing smoke and export checks complement these tests; they do not replace the selected checkpoint's final evaluation.

## Limits and unresolved competition results

The local implementation has no claim of new competitive training. Existing reported results belong to the Colab runs. IMPALA was stopped under the deadline after its selected reported 100k model still drifted; neither hyperparameter tuning nor extra steps has been proved to repair it. A 500k training target already used rollout/minibatch updates. Starting with a lower learning rate, comparing rollout sizes, checking exploration and running multiple seeds are possible later experiments, not known solutions.

The basic-CNN 400k checkpoint is the preferred reported candidate. Its different reported seed-10000 times still need pinned-weight reproducibility checks. Full export parity of the actual selected file and organizer-interface verification must precede claims of a ready, accepted submission.
