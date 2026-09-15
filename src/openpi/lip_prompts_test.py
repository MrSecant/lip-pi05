import pickle
from typing import ClassVar

import numpy as np

from openpi.lip_prompts import CUCUMBER_PROMPTS
from openpi.lip_prompts import PromptedLipDataset
from openpi.lip_prompts import sample_prompt
from openpi.lip_sampling import ReferenceStage2Sampler
from openpi.models import tokenizer
from openpi.training.config import get_config


class DummyDataset:
    split = "train"
    manifest: ClassVar[dict] = {"prompt": "old prompt"}

    def __len__(self):
        return 1031

    def identity(self, index):
        return {"pair_index": index + 100}

    def __getitem__(self, index):
        return {"prompt": "old prompt", "actions": np.full((2, 3), index), "state": np.array([index])}


def test_prompt_candidates_are_uniform_reproducible_and_epoch_dependent():
    values = [sample_prompt(CUCUMBER_PROMPTS, seed=42, epoch=1, pair_index=i) for i in range(6000)]
    counts = [values.count(prompt) for prompt in CUCUMBER_PROMPTS]
    assert all(1800 < count < 2200 for count in counts)
    assert values == [sample_prompt(CUCUMBER_PROMPTS, seed=42, epoch=1, pair_index=i) for i in range(6000)]
    other = [sample_prompt(CUCUMBER_PROMPTS, seed=42, epoch=2, pair_index=i) for i in range(100)]
    assert values[:100] != other


def test_prompt_wrapper_preserves_samples_and_worker_pickle():
    source = DummyDataset()
    dataset = PromptedLipDataset(source, train_prompts=CUCUMBER_PROMPTS, eval_prompt=CUCUMBER_PROMPTS[1])
    reopened = pickle.loads(pickle.dumps(dataset))
    for i in range(30):
        item = dataset[i, 3]
        np.testing.assert_array_equal(item["actions"], source[i]["actions"])
        np.testing.assert_array_equal(item["state"], source[i]["state"])
        assert item["prompt"] == reopened[i, 3]["prompt"]
    source.split = "val"
    assert {dataset[i, 7]["prompt"] for i in range(30)} == {CUCUMBER_PROMPTS[1]}


def test_resume_keeps_sample_order_and_random_language():
    sampler = ReferenceStage2Sampler(1031, 8, world_size=8, rank_batch_size=4, include_epoch=True)
    dataset = PromptedLipDataset(DummyDataset(), train_prompts=CUCUMBER_PROMPTS)
    first, second = list(sampler), list(sampler)
    legacy = ReferenceStage2Sampler(1031, 8, world_size=8, rank_batch_size=4)
    assert [index for index, _ in first] == list(legacy)
    assert all(epoch == 1 for _, epoch in first)
    assert all(epoch == 2 for _, epoch in second)
    sampler.seek(len(first) // 8 + 3)
    resumed = list(sampler)
    assert resumed == second[24:]
    assert [dataset[i]["prompt"] for i in resumed] == [dataset[i]["prompt"] for i in second[24:]]


def test_production_language_contract(monkeypatch):
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda length: None)
    config = get_config("pi05_lip_cucumber_100k")
    data = config.data.create(None, config.model)
    assert config.batch_size == 128
    assert config.fsdp_devices == 8
    assert data.lip_train_prompts == CUCUMBER_PROMPTS
    assert data.lip_eval_prompt == CUCUMBER_PROMPTS[1]
