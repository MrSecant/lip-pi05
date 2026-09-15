"""Reproduce the original 8-rank Stage2 epoch membership before rebatching."""

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


def reference_epoch_indices(count, epoch, world_size=8, rank_batch_size=32):
    if count < 1 or epoch < 1 or world_size < 1 or rank_batch_size < 1:
        raise ValueError("Reference sampler sizes and epoch must be positive")
    rank_steps = (count // world_size) // rank_batch_size
    if rank_steps == 0:
        raise ValueError("Dataset is smaller than one reference global batch")
    ranks = []
    for rank in range(world_size):
        sampler = DistributedSampler(range(count), world_size, rank, shuffle=True, drop_last=True, seed=0)
        sampler.set_epoch(epoch)
        ranks.append(
            np.asarray(list(sampler)[: rank_steps * rank_batch_size], dtype=np.int64).reshape(
                rank_steps, rank_batch_size
            )
        )
    return np.stack(ranks, axis=1).reshape(-1)


def reference_metric_indices(count, max_batches, reference_batch_size):
    if max_batches is None or count <= max_batches * reference_batch_size:
        return np.arange(count, dtype=np.int64)
    return torch.linspace(0, count - 1, steps=max_batches * reference_batch_size).round().to(torch.int64).numpy()


class ReferenceStage2Sampler(Sampler):
    def __init__(self, count, batch_size, *, world_size=8, rank_batch_size=32, include_epoch=False):
        if min(count, batch_size, world_size, rank_batch_size) < 1:
            raise ValueError("Sampler sizes must be positive")
        self.count = count
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank_batch_size = rank_batch_size
        self.include_epoch = include_epoch
        self.epoch_size = (count // world_size // rank_batch_size) * world_size * rank_batch_size
        if not self.epoch_size or self.epoch_size % batch_size:
            raise ValueError("JAX batch must divide the reference epoch sample count exactly")
        self.seek(0)

    def seek(self, step):
        if step < 0:
            raise ValueError("Step cannot be negative")
        epoch, offset = divmod(step, self.epoch_size // self.batch_size)
        self.epoch = epoch + 1
        self.offset = offset * self.batch_size

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        indices = reference_epoch_indices(self.count, self.epoch, self.world_size, self.rank_batch_size)
        epoch = self.epoch
        offset = self.offset
        self.epoch += 1
        self.offset = 0
        indices = indices[offset:].tolist()
        if self.include_epoch:
            return iter((index, epoch) for index in indices)
        return iter(indices)
