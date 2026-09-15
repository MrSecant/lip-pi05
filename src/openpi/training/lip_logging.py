"""Stage2-compatible local metrics with separate step and epoch axes."""

import json
import math
from pathlib import Path


def steps_per_epoch(cache_path, global_batch):
    manifest = json.loads((Path(cache_path) / "manifest.json").read_text())
    reference_batch = int(manifest["stage2_config"]["train"]["batch_size"])
    reference_global_batch = 8 * reference_batch
    count = manifest["split_counts"]["train"] // reference_global_batch * reference_global_batch
    if global_batch < 1 or count < global_batch or count % global_batch:
        raise ValueError("Global batch must divide the original Stage2 epoch sample count")
    return count // global_batch


def sampling_metric_key(solver, steps, solvers):
    return f"{solver}/{steps}_step" if len(set(solvers)) > 1 else f"{steps}_step"


class LipRunLogger:
    def __init__(self, directory, epoch_steps, *, resume_step=0, log_interval=20, writer=None, step_axis=False):
        if epoch_steps < 1 or log_interval < 1 or resume_step < 0:
            raise ValueError("Invalid logger step configuration")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch_steps = epoch_steps
        self.log_interval = log_interval
        self.step_axis = step_axis
        self.last_step = resume_step
        self.loss_sum = 0.0
        self.loss_count = 0
        snapshot = self.root / "lip_logging_states" / f"step_{resume_step:08d}.json"
        if resume_step and snapshot.exists():
            saved = json.loads(snapshot.read_text())
            if saved["step"] != resume_step or saved["epoch_steps"] != epoch_steps:
                raise ValueError("Logger snapshot does not match the restored training step")
            self.loss_sum, self.loss_count = saved["loss_sum"], saved["loss_count"]
        if writer is None:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(self.root / "tensorboard"))
        self.writer = writer
        self.stream = (self.root / "metrics.jsonl").open("a", buffering=1)
        self._write(dict(kind="session", resumed_step=resume_step, epoch_steps=epoch_steps))

    def _write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def record_train(self, step, metrics, learning_rate):
        values = {key: float(value) for key, value in metrics.items()}
        if step != self.last_step + 1:
            raise ValueError("Training metrics must follow the restored step consecutively")
        if not all(math.isfinite(value) for value in (*values.values(), learning_rate)):
            raise ValueError("Non-finite training metric")
        self.last_step = step
        self.loss_sum += values["loss"]
        self.loss_count += 1
        if step == 1 or step % self.log_interval == 0:
            tags = {"step/train_loss": values["loss"], "step/lr": float(learning_rate)}
            tags.update({f"step/{key}": value for key, value in values.items() if key != "loss"})
            for key, value in tags.items():
                self.writer.add_scalar(key, value, step)
            self._write(dict(kind="step", step=step, scalars=tags))
        if step % (self.log_interval if self.step_axis else self.epoch_steps) == 0:
            epoch = step // self.epoch_steps
            tag = "train/loss" if self.step_axis or self.loss_count == self.epoch_steps else "train/loss_partial"
            mean = self.loss_sum / self.loss_count
            self.writer.add_scalar(tag, mean, step if self.step_axis else epoch)
            self._write(dict(kind="interval" if self.step_axis else "epoch", step=step, epoch=epoch,
                samples_steps=self.loss_count, scalars={tag: mean}))
            self.loss_sum, self.loss_count = 0.0, 0
            self.writer.flush()

    def record_eval(self, step, scalars):
        epoch = max(1, math.ceil(step / self.epoch_steps))
        values = {key: float(value) for key, value in scalars.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Non-finite evaluation metric")
        canonical = {}
        for key, value in values.items():
            if key.startswith("flow_loss/"):
                split = key.removeprefix("flow_loss/")
                key = "train/flow_loss_eval_subset" if split == "train" else f"eval/{split}/loss"
            self.writer.add_scalar(key, value, step if self.step_axis else epoch)
            canonical[key] = value
        self._write(dict(kind="eval", step=step, epoch=epoch, scalars=canonical))
        self.writer.flush()

    def save_progress(self, step):
        if step != self.last_step:
            raise ValueError("Logger and checkpoint steps differ")
        directory = self.root / "lip_logging_states"
        directory.mkdir(exist_ok=True)
        target = directory / f"step_{step:08d}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(step=step, epoch_steps=self.epoch_steps,
            loss_sum=self.loss_sum, loss_count=self.loss_count), allow_nan=False))
        temporary.replace(target)
        self.stream.flush()
        self.writer.flush()

    def close(self):
        self.stream.close()
        self.writer.close()
