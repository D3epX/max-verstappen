"""Checkpoint regularly and select candidates using separate validation races."""
import json
from stable_baselines3.common.callbacks import BaseCallback
from .checkpoint import save_checkpoint
from .config import VALIDATION_SEEDS
from .evaluate import evaluate_tracks, summarize, selection_key


class RacingCallback(BaseCallback):
    """Counters count transitions across workers, not callback invocations."""
    def __init__(self, run_dir, config, run):
        super().__init__()
        self.run_dir, self.config, self.run = run_dir, config, run
        self.history_path = run_dir / "validation_history.json"
        self.history = json.loads(self.history_path.read_text()) if self.history_path.exists() else []
        self.score_path = run_dir / "best_validation_score.json"
        # A history entry alone is not proof its model was successfully saved.
        if self.score_path.exists() and (run_dir / "best_validation.zip").exists():
            best = json.loads(self.score_path.read_text())
            self.best_key = selection_key(best["summary"], best["tracks"])
        else:
            self.best_key = (float("inf"), float("inf"))

    def _on_training_start(self):
        self.next_save = (self.model.num_timesteps // self.run.save_every + 1) * self.run.save_every
        self.next_evaluation = (self.model.num_timesteps // self.run.evaluate_every + 1) * self.run.evaluate_every

    def _on_step(self):
        steps = self.model.num_timesteps
        if steps >= self.next_save:
            save_checkpoint(self.model, self.run_dir / "latest.zip")
            self.next_save += self.run.save_every
        if steps >= self.next_evaluation:
            rows = evaluate_tracks(self.model, self.config, VALIDATION_SEEDS[:5],
                                   max_episode_steps=self.run.max_episode_steps)
            summary = summarize(rows)
            item = {"timesteps": steps, "summary": summary, "tracks": rows}
            key = selection_key(summary, rows)
            if key < self.best_key:
                # Save before printing: notebook display must not prevent selection.
                save_checkpoint(self.model, self.run_dir / "best_validation.zip")
                self.score_path.write_text(json.dumps(item, indent=2))
                self.best_key = key
            self.history.append(item)
            self.history_path.write_text(json.dumps(self.history, indent=2))
            self.logger.record("validation/penalized_seconds", summary["mean_penalized_seconds"])
            self.logger.record("validation/completion_rate", summary["completion_rate"])
            self.next_evaluation += self.run.evaluate_every
            print(f"Validation at {steps:,}: {summary}", flush=True)
        return True
