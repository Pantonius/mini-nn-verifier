# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Download, convert, and cache the MNIST dataset as float64 .bin files."""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np


def prepare_mnist():
    """Prepare the full MNIST dataset, returning paths to cached .bin files.

    Downloads raw MNIST if needed, converts to float64 .bin, and caches
    everything under the shared cache directory.
    """
    cache_dir = _get_cache_dir() / "mnist"

    # Step 1: Download raw .gz files if needed
    raw_dir = cache_dir / "raw"
    _download_mnist(raw_dir)

    # Step 2: Convert to float64 .bin if needed
    if not (cache_dir / "train_images.bin").exists():
        _convert_to_bin(raw_dir, cache_dir)

    return {
        "train_images": str(cache_dir / "train_images.bin"),
        "train_labels": str(cache_dir / "train_labels.bin"),
        "test_images": str(cache_dir / "test_images.bin"),
        "test_labels": str(cache_dir / "test_labels.bin"),
    }


def prepare_batches(cache_dir, split, in_size, eval_batch_size):
    """Split a dataset into batch .bin files for eval, cached under batches_<N>/.

    Returns a sorted list of Path objects for the batch files.
    """
    cache_dir = Path(cache_dir)
    batch_dir = cache_dir / f"batches_{eval_batch_size}"

    # Return cached batches if they exist
    existing = sorted(batch_dir.glob(f"{split}_batch_*.bin"))
    if existing:
        return existing

    batch_dir.mkdir(parents=True, exist_ok=True)
    images = np.fromfile(cache_dir / f"{split}_images.bin", dtype=np.float64)
    num_samples = images.size // in_size
    images = images.reshape(num_samples, in_size)

    if num_samples % eval_batch_size != 0:
        raise ValueError(
            f"number of {split} samples ({num_samples}) is not divisible "
            f"by eval_batch_size ({eval_batch_size})"
        )

    num_batches = num_samples // eval_batch_size
    batch_paths = []
    for i in range(num_batches):
        batch = images[i * eval_batch_size : (i + 1) * eval_batch_size]
        path = batch_dir / f"{split}_batch_{i:03d}.bin"
        batch.tofile(path)
        batch_paths.append(path)
    return batch_paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _get_cache_dir():
    return Path(
        os.environ.get(
            "MININNVERIFIER_CACHE_DIR", Path.home() / ".cache" / "mininnverifier" / "datasets"
        )
    )


def _download_mnist(raw_dir):
    """Download raw MNIST .gz files if not already present."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in _FILES:
        filepath = raw_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(_BASE_URL + filename, filepath)
            print(f"Downloaded {filename}")


def _load_images(filepath):
    """Load MNIST images from a compressed .gz file as float64."""
    with gzip.open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        return images.astype(np.float64) / 255.0


def _load_labels(filepath, num_classes=10):
    """Load MNIST labels from a compressed .gz file as float64 one-hot."""
    with gzip.open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def _convert_to_bin(raw_dir, out_dir):
    """Convert raw MNIST .gz files to float64 .bin files."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_images = _load_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = _load_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = _load_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _load_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")

    train_images.tofile(out_dir / "train_images.bin")
    train_labels.tofile(out_dir / "train_labels.bin")
    test_images.tofile(out_dir / "test_images.bin")
    test_labels.tofile(out_dir / "test_labels.bin")
