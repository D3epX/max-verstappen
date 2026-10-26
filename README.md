# Max Verstappen — learning to race in four days

A beginner's CarRacing project built one function at a time, with the goal of learning to drive quickly on unseen tracks.

**Recommended approach:** train your own continuous-action PPO driver with four grayscale frames and a small CNN. Once that baseline works, compare a small IMPALA-style residual CNN. The choice is a practical recommendation for a beginner using Colab, not a claim of a proven competition winner.

## What exists today

`src/ppo/environment.py` implements track sampling and resets for separate training, validation, test and public seed groups. The other PPO modules, both CNN folders, and the two notebooks are empty placeholders to fill in together. No PPO driver has been trained yet.

## Set up in Google Colab

Choose **Runtime → Change runtime type → GPU**, then check the assigned hardware:

```python
import sys
import torch
print(sys.version)
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Upload this repository to your own GitHub, clone it into `/content`, and change into its directory. Alternatively upload the repository files directly using Colab's Files panel. `requirements.txt` must be in the current directory before running:

```python
%pip install -r requirements.txt
%pip check
```

Restart the runtime after installation if Colab requests it, then rerun the hardware check. The requirements allow a compatible preinstalled PyTorch to remain; do not blindly replace Colab's CUDA build with a CPU build. Python 3.12 is recommended. The core versions are Gymnasium 1.3.0 and Stable-Baselines3 2.9.0; change the Gymnasium pin to the organizer's exact version before final training. SB3 2.9 requires PyTorch >=2.8. [SB3 installation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html), [release notes](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html).

Run this installation check. It uses random actions and does **not** train a driver:

```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch

print("Gymnasium", gym.__version__, "SB3", sb3.__version__, "Torch", torch.__version__)
env = gym.make("CarRacing-v3", continuous=True, max_episode_steps=3000)
try:
    obs, info = env.reset(seed=202)
    assert obs.shape == (96, 96, 3) and obs.dtype == np.uint8
    assert env.action_space.shape == (3,)
    assert env.spec.max_episode_steps == 3000
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    print("Environment installation check passed")
finally:
    env.close()
```

For images inside a notebook, display the observation with `matplotlib.pyplot.imshow(obs)`. Training does not need a desktop window or `render_mode="human"`. Video encoding is included through `imageio[ffmpeg]`. If Box2D installation attempts a source build and reports missing SWIG, install `swig` first in a separate `%pip install "swig==4.*"` cell, then retry. A missing compiler may require Colab's `!apt-get update` and `!apt-get install -y build-essential`.

Mount Drive for backups:

```python
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
backup_dir = Path("/content/drive/MyDrive/max-verstappen")
backup_dir.mkdir(parents=True, exist_ok=True)
```

Train and write frequent logs on `/content`, then copy completed checkpoint files to Drive every 10–15 minutes. Save model/optimizer state, configuration, package versions, total steps, seed settings and validation results. Confirm a saved checkpoint can reload before the first long run. Resume SB3 training with `PPO.load(...)` and `learn(..., reset_num_timesteps=False)`; a normal SB3 save does not capture the simulator's exact state or guarantee a bit-identical continuation.

Colab GPUs and session lengths are not guaranteed. Measure actual throughput and plan short, resumable runs; do not assume four uninterrupted training days. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Local CPU setup

Use your computer for learning, debugging and short checks, and Colab for the longer experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

GPU availability, full dependency installation and Colab runtime execution must be verified in the actual training environment. Before submission, record its exact installed package versions, for example with `python -m pip freeze > requirements-lock.txt` in a dedicated environment. The current requirements pin the core RL/environment packages and bound the supporting dependencies; they are not a complete lockfile.

Validation performed on this update: requirements resolved successfully in a dry run for Python 3.12; documentation links, code fences and whitespace were checked. The full stack has not been installed or run here.
