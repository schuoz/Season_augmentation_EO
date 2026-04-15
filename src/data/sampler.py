from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool = True,
):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler
