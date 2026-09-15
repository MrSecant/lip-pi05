import json

import pytest

from openpi.training.lip_logging import LipRunLogger
from openpi.training.lip_logging import sampling_metric_key
from openpi.training.lip_logging import steps_per_epoch


class Writer:
    def __init__(self):
        self.events = []

    def add_scalar(self, tag, value, step):
        self.events.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


def test_epoch_steps_preserve_old_sampler(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "stage2_config": {"train": {"batch_size": 32}}, "split_counts": {"train": 652194}}))
    assert steps_per_epoch(tmp_path, 64) == 10188
    with pytest.raises(ValueError):
        steps_per_epoch(tmp_path, 5)


def test_metric_tags_match_stage2():
    assert sampling_metric_key("heun", 32, ["heun"]) == "32_step"
    assert sampling_metric_key("euler", 8, ["heun", "euler"]) == "euler/8_step"


def test_epoch_mean_resumes_from_checkpoint_snapshot(tmp_path):
    first = LipRunLogger(tmp_path, 4, writer=Writer(), log_interval=1)
    for step in (1, 2, 3):
        first.record_train(step, {"loss": step, "grad_norm": 1.0}, 1e-5)
    first.save_progress(3)
    first.close()
    writer = Writer()
    resumed = LipRunLogger(tmp_path, 4, resume_step=3, writer=writer, log_interval=1)
    resumed.record_train(4, {"loss": 4.0}, 1e-5)
    resumed.close()
    assert ("train/loss", 2.5, 1) in writer.events
    assert ("step/train_loss", 4.0, 4) in writer.events


def test_missing_snapshot_does_not_fake_full_epoch(tmp_path):
    writer = Writer()
    logger = LipRunLogger(tmp_path, 4, resume_step=3, writer=writer)
    logger.record_train(4, {"loss": 2.0}, 1e-5)
    logger.close()
    assert ("train/loss_partial", 2.0, 1) in writer.events
    assert not any(tag == "train/loss" for tag, _, _ in writer.events)


def test_eval_axes_and_legacy_flow_alias(tmp_path):
    writer = Writer()
    logger = LipRunLogger(tmp_path, 4, writer=writer)
    logger.record_eval(8, {"flow_loss/validation": 1.0, "decoded_oracle/eval/validation/action_l1_unnormalized": 0.1})
    logger.close()
    assert ("eval/validation/loss", 1.0, 2) in writer.events
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["scalars"]["eval/validation/loss"] == 1.0


def test_invalid_steps_and_nan_rejected(tmp_path):
    logger = LipRunLogger(tmp_path, 4, writer=Writer())
    with pytest.raises(ValueError, match="consecutively"):
        logger.record_train(2, {"loss": 1.0}, 1e-5)
    with pytest.raises(ValueError, match="Non-finite"):
        logger.record_train(1, {"loss": float("nan")}, 1e-5)
    logger.close()


def test_real_tensorboard_events(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logger = LipRunLogger(tmp_path, 2, log_interval=1)
    logger.record_train(1, {"loss": 1.0, "grad_norm": 2.0}, 1e-5)
    logger.record_train(2, {"loss": 3.0, "grad_norm": 1.0}, 1e-5)
    logger.record_eval(2, {"latent_metrics/eval/val/heun/8_step/l1": 0.3})
    logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("train/loss")[0].value == 2.0
    assert events.Scalars("train/loss")[0].step == 1
    assert events.Scalars("step/train_loss")[-1].step == 2
    assert events.Scalars("latent_metrics/eval/val/heun/8_step/l1")[0].step == 1


def test_step_based_logging(tmp_path):
    writer = Writer()
    logger = LipRunLogger(tmp_path, 10188, log_interval=2, writer=writer, step_axis=True)
    logger.record_train(1, {"loss": 1.0}, 5e-8)
    logger.record_train(2, {"loss": 3.0}, 1e-7)
    logger.record_eval(5000, {"eval/val/loss": 0.5})
    logger.close()
    assert ("train/loss", 2.0, 2) in writer.events
    assert ("eval/val/loss", 0.5, 5000) in writer.events
