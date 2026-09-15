import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from openpi.lip_sampling import ReferenceStage2Sampler
from openpi.lip_sampling import reference_epoch_indices
from openpi.lip_sampling import reference_metric_indices


def test_exact_original_rank_batches():
    count, world, batch = 1031, 8, 4
    combined = reference_epoch_indices(count, 1, world, batch).reshape(-1, world, batch)
    for rank in range(world):
        sampler = DistributedSampler(range(count), world, rank, shuffle=True, drop_last=True)
        sampler.set_epoch(1)
        expected = list(sampler)[: combined.shape[0] * batch]
        np.testing.assert_array_equal(combined[:, rank].reshape(-1), expected)


def test_resume_matches_uninterrupted_sequence():
    sampler = ReferenceStage2Sampler(1031, 8, world_size=8, rank_batch_size=4)
    first = list(sampler)
    second = list(sampler)
    sampler.seek(len(first) // 8 + 3)
    assert list(sampler) == second[24:]


def test_metric_subset_rounds_like_original_torch():
    expected = torch.linspace(0, 652194 - 1, steps=128).round().to(torch.int64).numpy()
    np.testing.assert_array_equal(reference_metric_indices(652194, 4, 32), expected)
