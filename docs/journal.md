# Development journal

## 2026-10-23 — planning and repository review

This entry records planning only. No driver has been trained and no race results have been measured.

- Initial repository: challenge guidelines, one-line README, empty requirements file, and empty `src/`.
- Starting knowledge: learning PyTorch, CNNs and reinforcement learning from the beginning.
- Compute plan: Google Colab; no local GPU available.
- Proposed baseline: continuous-action PPO with four grayscale frames and a compact CNN, trained from scratch on varied track seeds.
- Proposed optional comparison: a small IMPALA-style residual CNN with the same PPO/evaluation setup.
- Added a four-day roadmap, research notes and dependency specification with coding-agent assistance.
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
