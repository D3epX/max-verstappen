# Run the CarRacing notebooks in Colab

The original baseline is `ppo_cnn.ipynb`. For the current comparison, use **`ppo_cnn_gsde_recovery.ipynb` and the updated `ppo_impala_cnn.ipynb`**: both use gSDE, resampling every four decisions, noise log-scale initialization -2 and no entropy bonus. The original basic-CNN notebook uses different exploration settings. All are self-contained; upload the notebook without cloning the repository. They use Stable-Baselines3's PPO implementation and define their CNNs, environment wrappers, checkpointing and evaluation inside the notebook.

## First run

1. Open [Google Colab](https://colab.research.google.com/), select **File → Upload notebook**, and choose `ppo_cnn.ipynb`.
2. Select **Runtime → Change runtime type → GPU**, then an available NVIDIA GPU such as T4. Save and connect. GPU availability varies; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
3. Run the installation cell. Restart the session only if Colab requests it, then run the import cell. Confirm it prints `Device: cuda` before a long run.
4. Run the Drive cell and authorize your own Drive connection. Leave `USE_DRIVE=True` so checkpoints survive a runtime disconnect.
5. Run the remaining cells in order. The short training check runs about 1,024 policy transitions, saves `latest.zip`, and checks reloaded actions. This is a pipeline check, not enough to learn racing.
6. In section 12, set `RUN_LONG_TRAINING=True`. Start with `TARGET_TOTAL_TIMESTEPS=50_000` to measure performance. Later increase the cumulative target to `250_000`, `500_000`, or `1_000_000` as time allows. These counts do not guarantee convergence.
7. After a training segment, rerun section 14 to select the evaluation checkpoint. Set `RUN_FULL_VALIDATION=True` to evaluate all 20 validation tracks. Set `RECORD_VIDEO=True` in section 15 to watch one validation race inline.
8. Leave `RUN_FINAL_TEST=False` until you have selected the architecture and settings. Keep the final test separate from model tuning.

Long training, full validation, video recording, public-track reporting, final testing and weight export each have their own switches. Running all cells with their initial settings performs the short check but does not start long training or evaluate the final test.

## Where files go

Check Google Drive under:

```text
MyDrive/max-verstappen/basic_cnn/run_01/
MyDrive/max-verstappen/basic_cnn/gsde_recovery_01/
MyDrive/max-verstappen/impala_cnn/gsde_comparison_01/
```

`latest.zip` resumes SB3 training, including optimizer state. `best_validation.zip` is selected by periodic five-track validation using DNF-penalized time. Configuration, installed package versions, validation reports and optional videos are saved alongside them. Back up TensorBoard logs using section 13 after a training segment.

The optional `.pth` export contains policy weights and settings. It still needs the matching architecture, preprocessing and an organizer-compatible inference adapter; it is not a complete official submission loader.

## Resume after a disconnect

Reconnect to a GPU runtime, rerun installation/imports, mount Drive, and run the configuration and class/function definition cells. Keep the same experiment name and settings. In section 9 set:

```python
RESUME_FROM = str(RUN_DIR / "latest.zip")
```

Then create/load the model, define the callback functions in section 10, skip the smoke update with `RUN_SMOKE=False` if unnecessary, and run section 12 with a larger cumulative target. The checkpoint does not restore the exact simulator state or guarantee identical continuation.

## Run the second architecture

Save the first run and finish its runtime before training the other notebook. Upload the updated `ppo_impala_cnn.ipynb` and repeat the same setup. Leave `RESUME_FROM=""` for the first IMPALA run: it starts fresh in `impala_cnn/gsde_comparison_01`. Do not load basic-CNN weights or older ordinary-Gaussian IMPALA checkpoints into this experiment. Compare with the basic-CNN **recovery** notebook using validation completion rate, mean penalized time, transitions and measured training time. Their configurations match except for the encoder. The first IMPALA cumulative target defaults to 100,000, with long training still opt-in; raise it toward a comparable budget if time allows. Both smoke and long training use `progress_bar=False` to avoid the observed Rich/Colab recursive-output failure. Section 15a now offers a deterministic control diagnostic compatible with gSDE.

## Diagnose circling before restarting training

Both notebooks include an optional section **15a** comparing three training-pool tracks and three validation tracks with deterministic and sampled actions. It records the checkpoint count, raw Gaussian means, exploration spread, clipping probabilities, rewards and completion. Set `RUN_CIRCLING_DIAGNOSTICS=True` to use it when training is idle. Twelve episodes can take several minutes. It does not learn from validation tracks or touch final-test tracks.

For an already-running older notebook, keep that runtime. Upload `circling_diagnostics.py` using Colab's Files panel, then run a new code cell:

```python
%run /content/circling_diagnostics.py
diagnostic_model = PPO.load(str(RUN_DIR / "latest.zip"), device=DEVICE)
circling_report = diagnose_circling(
    diagnostic_model, build_vec_env,
    TRAINING_SEEDS, VALIDATION_SEEDS, RUN_DIR,
)
```

Share the JSON report before choosing a change. A short training run may not have visited all training-pool tracks, and one rollout per track/action mode is a diagnostic rather than proof of causation. A proposed lower initial standard deviation is a fresh-run experiment, not a confirmed repair. Loading an old checkpoint restores its learned standard deviation regardless of a new initialization setting.

## Recovery experiment after the 50k circling checkpoint

`ppo_cnn_gsde_recovery.ipynb` is a separate fresh-run experiment. It keeps the basic CNN and environment pipeline, but tries state-dependent exploration (gSDE), a four-decision noise resampling interval, noise-weight initialization at -2, and no entropy bonus. These choices are motivated by the [SB3 maintainers' CarRacing recipe](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml); the notebook does not reproduce the complete recipe or claim its results.

Preserve `basic_cnn/run_01/latest.zip`. Upload the recovery notebook to Colab separately, choose the GPU, mount Drive and use its default new folder, `basic_cnn/gsde_recovery_01`. Do not resume the old diagonal-Gaussian weights into this different exploration parameterization. The normal section 11 short check and section 12 opt-in training workflow apply. Its local smoke test passed; better driving still needs longer training and validation.

The original 50k model is a failed early baseline, not a completed racing agent. Its checkpoint contains only 36 completed episodes. Continuing the unchanged baseline remains a useful control if compute allows. Use comparable budgets and report the actual held-out results rather than assume the recovery configuration will win.

The saved baseline's measured results are in [baseline_50176_diagnostics.json](results/baseline_50176_diagnostics.json): two training-pool tracks and two validation tracks, each tested with deterministic and sampled actions, with zero finishes in eight full episodes. This diagnoses a poorly learned turning policy; it does not isolate a unique learning cause or establish that the recovery settings work.

`load_policy_weights.py` supplies an inference-only alternative for inspecting the original notebooks' diagonal-Gaussian checkpoints across Python minor versions. It reconstructs the provided local encoder and loads tensor weights without unpickling notebook code. It is not a training-resume loader and does not support the gSDE recovery policy.

## Watch the selected trained agent with Python

`src/watch_agent.py` supports the basic CNN, IMPALA CNN, and gSDE recovery checkpoints. It is separate from `src/set_up.py`, which drives with random sampled actions. The viewer reconstructs the known policy architecture and loads tensor weights, allowing inference even when Colab and the laptop use different Python minor versions. It does not train or restore an optimizer.

After training, use section 14 to select `best_validation.zip` and inspect its validation results. Section 15 records that selected policy in Colab. Periodic validation starts at 100,000 transitions with the current defaults, so earlier runs may only have `latest.zip`. A file called "best" is only best among the checked checkpoints, and may still fail every race.

In Google Drive, open `MyDrive/max-verstappen/basic_cnn/gsde_recovery_01`, download `best_validation.zip` to Downloads, then run locally:

```bash
cd /home/dahmane/dev/max-verstappen
.venv/bin/python src/watch_agent.py --model /home/dahmane/Downloads/best_validation.zip --seed 10000
```

This opens a game window and uses deterministic policy actions. If the downloaded filename differs, use that actual path. Before best-validation selection exists, explicitly substitute the desired `latest.zip` path. The previous failing baseline's `latest.zip` will still show that old policy, not the recovery model.

To record an MP4 instead of opening a window:

```bash
cd /home/dahmane/dev/max-verstappen
.venv/bin/python src/watch_agent.py --model /home/dahmane/Downloads/best_validation.zip --seed 10000 --video /home/dahmane/Downloads/trained_car.mp4
```

The viewer prints completion, raw simulator steps, lap time, penalized time and reward. One track/video is not a complete evaluation. Leave the default 3,000-step limit for challenge-style checks; its shorter-step option is only for debugging.

Local checks passed for loading the uploaded original checkpoint, the short-check IMPALA and gSDE checkpoints, matching the original policy's initial actions, deterministic stepping, command-line parsing, and MP4 writing under shortened test episodes. A desktop window has not been tested through the headless tool environment.

## Optional public-track fine-tuning

Use `ppo_cnn_public_finetune.ipynb` only after permission to train on public seeds. It reads the selected 400k basic-CNN ZIP from `basic_cnn/gsde_recovery_01/best_validation.zip`, hashes/preserves it and trains a separate copy in `basic_cnn/public_mix_from_400k_01`. Do not use the 500,526 latest model as the parent.

The sampler chooses ordinary training roads on 80% of episode resets and one of seeds 202/42/203 on 20%; validation/test remain outside training. Learning rate changes to `1e-4` with the actual model schedule refreshed. This is an experiment, not a promised improvement.

Run setup and definitions through section 10a. It evaluates the protected baseline on all three public roads and ten validation roads. Baseline caching includes parent identity and runtime/device context. Keep `RUN_SMOKE=False`. In section 12 set `RUN_LONG_TRAINING=True` and leave `ADDITIONAL_TIMESTEPS=10_000`. The cumulative target is parent+10k, roughly 410k; complete rollouts can overshoot. Afterward run section 14 for the same public/validation audit. Negative time deltas mean improvement; preserve the parent if the copy deteriorates. Nothing automatically publishes a release or replaces the original.

To resume, keep the same fine-tune folder and sampling settings and set `RESUME_FROM=str(RUN_DIR / "latest.zip")` in section 9. The ordinary local training CLI rejects mixed-sampling resumes so it cannot silently return to uniform sampling. Use this notebook for its mixed sampler.
