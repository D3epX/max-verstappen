# Model v1 — PPO + basic CNN

**Baseline release:** continuous-action PPO with a three-layer CNN and gSDE, selected at **400,000 training transitions**. The final training run reached 500,526, but its last checkpoint scored worse on the shared validation comparison.

## Results

On ten validation tracks, seeds 10000–10009, with DNF scored as 70 seconds:

- **Selected 400k:** 6/10 finishes, mean penalized time **52.948 s**, mean successful lap time **41.580 s**.
- **Latest 500,526:** 4/10 finishes, mean penalized time **61.758 s**, mean successful lap time **49.395 s**.

These figures are from the supplied Colab evaluation output. Complete per-track results and provenance are in [reported_validation_comparison.json](https://github.com/D3epX/max-verstappen/blob/model-v1/results/reported_validation_comparison.json). The ten validation tracks were used for model selection, not as an untouched final test. Public-seed 202/42/203 and final-test results are not yet supplied. No public-track or mystery-track performance is claimed.

An additional six-episode check is recorded separately in [additional_reported_checks.json](https://github.com/D3epX/max-verstappen/blob/model-v1/results/additional_reported_checks.json): three training-pool timeouts, two validation finishes at 57.78 s and 57.50 s, and one validation timeout. Its checkpoint identity was not supplied; it is not attributed to the selected model.

## Model

- Input: raw `(96, 96, 3)` RGB `uint8` observation, one call per simulator step.
- Preprocessing: grayscale, bilinear resize to 84×84, four decision-time frames, zero-padded reset history.
- Encoder: three convolutions (32, 64, 64 channels), then a 256-feature projection.
- Actor/critic: shared image features and separate 64-unit Tanh heads.
- Training: PPO clipped objective, GAE, Adam and gSDE. Noise is resampled every four decisions; deterministic inference uses mean controls.
- Output: NumPy `[steering, gas, brake]` in `[-1,1] × [0,1] × [0,1]`.
- Action repeat: two simulator steps, handled by the exported Driver.

The IMPALA-style residual encoder remains in source as an alternative. Its reported selected 100k model still drifted in the shown attempt, and that experiment was stopped under the deadline. Neither tuning nor additional timesteps is presented as a verified repair.

## Download and load the weights

The exported asset is [carracing_basic_cnn_400000_weights.pth](https://github.com/D3epX/max-verstappen/releases/download/model-v1/carracing_basic_cnn_400000_weights.pth) (**3,667,229 bytes**). It contains the selected policy tensors, matching configuration, 400,000-transition counter and source-checkpoint fingerprint. It is a tensor bundle rather than a standalone pickled model.

Install the source and dependencies, then download the PTH asset into the repository's `submission/` folder:

```bash
git clone --branch model-v1 https://github.com/D3epX/max-verstappen.git
cd max-verstappen
uv venv --python 3.12
uv pip install -r requirements.txt
mkdir -p submission
curl -fL https://github.com/D3epX/max-verstappen/releases/download/model-v1/carracing_basic_cnn_400000_weights.pth -o submission/carracing_basic_cnn_400000_weights.pth
```

Run the following with `.venv/bin/python` from the repository root:

```python
from src.ppo.driver import load_driver

model = load_driver("submission/carracing_basic_cnn_400000_weights.pth")
model.eval()
model.reset()
action = model(observation)  # raw (96, 96, 3) RGB uint8 -> NumPy controls
```

Call `model.reset()` after each environment reset. Call the driver once per raw simulator step; do not add a second action-repeat wrapper. It uses supplied pixels and has no access to hidden game state. Keep the original ZIP for training continuation: this inference bundle does not export optimizer state.

## Artifact identity

- Source checkpoint: `selected_400000(1).zip`, verified native counter **400,000**.
- Source SHA-256: `903c0d28ac56b99d5adaaacdd38b2dcf7f6ff59d544ce09276886276492dde51`.
- Exported PTH SHA-256: `7b4854b3e0e8efe6f8087c755131ebab116704402a2cf98cf3f280519b87b6cb`.

Release assets include the PTH inference bundle, `selected_400000.zip` for resuming training, `SHA256SUMS`, dependency list and evaluation reports. Model binaries are distributed as release assets rather than source-control files. Verify downloaded assets with `sha256sum -c SHA256SUMS` after downloading all listed files into one directory.

**Verification:** the selected 400k weights have been exported and strict-loaded with the local inference adapter. Full-episode action/preprocessing parity verification passed against the source ZIP on seed 10000: both finished at 1,818 raw steps (36.36 s), with matching deterministic controls, reward sums and ending flags. This local check agrees with the earlier 36.36 s video; it does not reproduce the later reported 23.48 s run. This does not certify organizer acceptance or say the model has been submitted.

The earlier seed-10000 video reported 36.36 s while the later Colab comparison reported 23.48 s. The exact file and runtime must match before comparing those timings. This release includes the verified 400k checkpoint; no additional training results are claimed.
