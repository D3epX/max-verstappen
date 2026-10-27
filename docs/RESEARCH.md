# Research and architecture decision

Research checked: **23 October 2026**. These are selected relevant papers, not an exhaustive survey. No source below establishes a fastest driver under this competition's precise seeds, scoring, compute budget and interview constraints. “Best for this project” is an engineering judgment that needs an experiment.

## Decision

Use **PPO + four stacked grayscale images + a small CNN** first. Train the weights yourself. If that works and compute remains, test **PPO + a small IMPALA-style residual CNN** under the same wall-clock budget and validation protocol. You have no local GPU and plan to use Colab, so development cost and interrupted sessions matter.

PPO is the learning algorithm; a CNN is the image-processing network inside it. A newer algorithm and a deeper network are separate choices. Better average rewards on unrelated benchmarks do not establish faster CarRacing laps.

## Papers worth knowing

### PPO — Schulman et al., 2017

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). PPO gathers experience with its current policy, then makes several learning passes over that experience. Its clipped objective discourages excessively large policy changes. This is the one algorithm paper to study during the four days: abstract, algorithm outline, and clipped-objective explanation. Use the maintained SB3 implementation rather than building the optimizer from scratch.

### IMPALA — Espeholt et al., 2018

[IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561). The useful architecture idea here is a compact residual image encoder. Borrowing that encoder for PPO does not require implementing IMPALA's distributed learning algorithm or V-trace.

### DrQ-v2 — Yarats et al., 2021 preprint / 2022 publication

[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/abs/2107.09645). An off-policy visual-control method that uses data augmentation and reuses experience. It is a relevant alternative when environment interactions are expensive. Reproducing it also means handling replay storage, augmentations and additional learning components. Keep it as a future project; do not switch to it late in this four-day attempt.

### TD-MPC2 — Hansen et al., ICLR 2024

[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html). Learns a compact dynamics model and plans actions using it. It is relevant continuous-control research, but its results are not evidence of winning this race. Planning and dynamics-model training add concepts you would need to debug and explain.

### DreamerV3 — Hafner et al., Nature 2025

[Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2). Learns a world model and improves a policy through imagined experience across many tasks. Its PPO comparison uses an IMPALA network, supporting that encoder as a credible candidate to test. This is not a CarRacing lap-time comparison. A world-model implementation would add substantial learning and compute demands to your deadline.

### Dreamer 4 — Hafner, Yan and Lillicrap, September 2025 preprint

[Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527), [authors' project](https://danijar.com/project/dreamer4/). Trains behavior inside a learned world model and demonstrates offline Minecraft learning. It is newer research than DreamerV3, but solves a different data/setup problem. Do not infer that a large transformer world model is the best architecture for a small racing competition. No pretrained racing weights should enter your submission.

### Newt — Hansen et al., November 2025 preprint / ICLR 2026

[Learning Massively Multitask World Models for Continuous Control](https://arxiv.org/abs/2511.19584), [ICLR paper](https://openreview.net/pdf/373f3e2317f8e865dc41ea3096fa3646726a67b8.pdf). Combines demonstration pretraining and online learning for a language-conditioned multitask world model. Its broad multitask setup is valuable research context, but does not match learning this single pixel-based driving task in four days.

### OG-SPR — Liu et al., August 2026 preprint

[Observation-Grounded Self-Predictive Reinforcement Learning for Visual Continuous Control](https://arxiv.org/abs/2608.05989). Combines future latent prediction and next-observation prediction to improve visual representations. Its abstract reports experiments on 28 DeepMind Control Suite tasks, not this challenge. Treat its claims as preprint results and a future reading direction, not a validated competition recipe.

## The stronger encoder to test

A small IMPALA-style candidate, trained from scratch:

```text
Input: batch × 4 × 84 × 84, floating point scaled to [0,1]
Stage 1: Conv 3×3, 16 channels → MaxPool 3×3/stride 2 → 2 residual blocks
Stage 2: Conv 3×3, 32 channels → MaxPool 3×3/stride 2 → 2 residual blocks
Stage 3: Conv 3×3, 32 channels → MaxPool 3×3/stride 2 → 2 residual blocks
ReLU → Flatten → Linear to 256 → ReLU
Actor: 256 → 64 → 3 Gaussian means, plus 3 learned log standard deviations
Critic: 256 → 64 → 1 estimated return
```

Use padding 1 for the convolutions and pool operations. A residual block can be `x + Conv3x3(ReLU(Conv3x3(ReLU(x))))`, keeping spatial size and channel count constant. A skip connection preserves a direct path for information and gradients. It does not guarantee better performance. This is a proposed small variant, not a claim to reproduce any paper's exact experimental configuration.

Use it as an SB3 custom feature extractor so PPO, action handling and evaluation stay the same. Start without BatchNorm, dropout, an LSTM, attention or a transformer. Frame stacking already gives a short motion history. Memory mechanisms are worth considering only after videos identify a failure requiring longer history.

Compare both **equal wall-clock budgets** (important on Colab) and the number of collected transitions. Log completion rate and DNF-penalized lap times on held-out seeds. A bigger network that halves collection speed may lose within your deadline. Keep the simple CNN if the residual encoder fails to improve those measurements.

## Read only what you need now

Spend about 30–45 minutes total on paper abstracts and diagrams after learning the basics. Spend the rest of your time implementing small pieces, checking them, running controlled experiments and explaining results. Full paper reproduction is a later learning goal.
