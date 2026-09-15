"""Exercise sharded evaluation and exclusion of padded tail samples."""
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import lip_evaluation as evaluation
from openpi.training import sharding


@pytest.mark.parametrize("step,expected", [(0, False), (1, False), (4999, False),
    (5000, True), (10000, True), (10188, False), (30000, True),
    (50000, True), (80000, True), (99999, False), (100000, True)])
def test_step_eval_schedule(step, expected):
    from openpi.training.config import get_config
    assert evaluation.evaluation_due(get_config("pi05_lip_cucumber_100k"), step) is expected


def test_official_pi05_optimization_recipe():
    from openpi.training.config import get_config
    ours = get_config("pi05_lip_cucumber_100k")
    official = get_config("pi05_libero")
    assert ours.lr_schedule == official.lr_schedule
    assert ours.optimizer == official.optimizer
    assert ours.ema_decay == official.ema_decay
    assert ours.num_train_steps == 100000
    assert ours.batch_size == 128 and ours.fsdp_devices == 8
    assert ours.lip_eval_every_steps == ours.save_interval == ours.keep_period == 5000
    assert ours.lip_eval_solvers == ("euler",)
    assert ours.lip_eval_sampling_steps == (8, 16)
    assert not ours.lip_eval_on_resume
    lr = ours.lr_schedule.create()
    assert float(lr(0)) == pytest.approx(5e-5 / 10001)
    for step in (10000, 30000, 50000, 80000, 100000):
        assert float(lr(step)) == pytest.approx(5e-5)


class TinyModel(nnx.Module):
    def __init__(self):
        self.max_token_len = 32
        self.weight = nnx.Param(jnp.zeros((128,)))

    def compute_loss(self, key, observation, target):
        return target[:, :, 0] + self.weight.value[0]

    def sample_actions(self, key, observation, *, noise, num_steps, solver):
        return noise + self.weight.value


@pytest.mark.parametrize("prompt", [None, "Peel the cucumber, then place the slice in the cup."])
@pytest.mark.parametrize("override", [False, True])
def test_sharded_eval_excludes_padded_tail(tmp_path, monkeypatch, prompt, override):
    batch_size = jax.device_count()
    counts = {"train": batch_size + 1, "val": 2 * batch_size + 1}
    cache = tmp_path / "cache"
    cache.mkdir()
    settings = {"max_batches": None, "sampling_steps": [1], "solvers": ["euler"]}
    manifest = {
        "rows_sha256": "test", "stage1_sha256": "test", "split_names": list(counts),
        "stage2_config": {
            "evaluation": {"target_metrics": {"train": settings, "eval": settings}},
            "train": {"batch_size": 32}, "model": {"sampling": {"solver": "euler"}},
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    reads = []

    class Dataset:
        def __init__(self, root, split, **kwargs):
            self.count = counts[split]

        def __len__(self):
            return self.count

        def identity(self, index):
            return {"pair_index": index}

        def __getitem__(self, index):
            reads.append(index)
            return {"actions": np.full((32, 128), index, dtype=np.float32)}

    class Observation:
        @staticmethod
        def from_dict(batch):
            return batch["actions"]

    monkeypatch.setattr(evaluation, "LipCacheDataset", Dataset)
    monkeypatch.setattr(evaluation, "Observation", Observation)
    monkeypatch.setattr(evaluation, "PaligemmaTokenizer", lambda length: None)
    def tokenize(row):
        if prompt is not None:
            assert row["prompt"] == prompt
            row = {key: value for key, value in row.items() if key != "prompt"}
        return row

    monkeypatch.setattr(evaluation, "TokenizePrompt", lambda *args, **kwargs: tokenize)
    output = evaluation.evaluate(TinyModel(), cache, tmp_path / "output",
        batch_size=batch_size, mesh=sharding.make_mesh(batch_size), prompt=prompt,
        solvers_override=("euler",) if override else (), sampling_steps_override=(8, 16) if override else ())
    assert len(reads) == sum(counts.values())
    report = json.loads((output / "predictions.json").read_text())
    assert report["prompt"] == prompt
    assert {g["steps"] for g in report["groups"]} == ({8, 16} if override else {1})
    assert {g["solver"] for g in report["groups"]} == {"euler"}
    for split, count in counts.items():
        assert report["flow_losses"][split] == (count - 1) / 2
        group = next(group for group in report["groups"] if group["split"] == split)
        predictions = np.load(output / group["predictions"])
        assert predictions.shape == (count, 32, 128)
        assert np.isfinite(predictions).all()
        np.testing.assert_array_equal(np.load(output / group["pair_ids"]), np.arange(count))
        # The mock sampler ignores num_steps, so equal results prove identical initial noise.
        for other in (g for g in report["groups"] if g["split"] == split):
            np.testing.assert_array_equal(predictions, np.load(output / other["predictions"]))
