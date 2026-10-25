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
