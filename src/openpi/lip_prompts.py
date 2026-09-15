"""Reproducible language augmentation without changing cached sample alignment."""

import numpy as np

CUCUMBER_PROMPTS = (
    "Peel the cucumber and put the slice into the cup",
    "Peel the cucumber, then place the slice in the cup.",
    "After peeling the cucumber, put the cucumber slice into the cup",
)


def sample_prompt(prompts, *, seed, epoch, pair_index):
    if not prompts or min(seed, epoch, pair_index) < 0:
        raise ValueError("Require prompts and nonnegative seed, epoch, and pair index")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, pair_index]))
    return prompts[int(rng.integers(len(prompts)))]


class PromptedLipDataset:
    def __init__(self, dataset, *, train_prompts=(), eval_prompt=None, seed=42):
        self.dataset = dataset
        self.train_prompts = tuple(train_prompts)
        self.eval_prompt = eval_prompt
        self.seed = seed

    @property
    def manifest(self):
        return self.dataset.manifest

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 1)
        sample = dict(self.dataset[index])
        if self.dataset.split == "train" and self.train_prompts:
            sample["prompt"] = sample_prompt(
                self.train_prompts, seed=self.seed, epoch=epoch,
                pair_index=int(self.dataset.identity(index)["pair_index"]),
            )
        elif self.eval_prompt is not None:
            sample["prompt"] = self.eval_prompt
        return sample
