# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .mnist import prepare_mnist

DATASETS = {"mnist": prepare_mnist}


def prepare_dataset(source_dataset):
    """Download/prepare a dataset and return absolute paths to .bin files.

    source_dataset is a dataset name string (e.g. "mnist").
    """
    if source_dataset not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset: {source_dataset!r}. Available: {available}")
    return DATASETS[source_dataset]()
