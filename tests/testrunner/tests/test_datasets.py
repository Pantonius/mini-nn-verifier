# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Tests for testrunner.datasets (mnist, prepare_dataset, prepare_batches)."""

import gzip
import io
import os
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from testrunner.datasets import DATASETS, prepare_dataset
from testrunner.datasets.mnist import (
    _convert_to_bin,
    _download_mnist,
    _get_cache_dir,
    _load_images,
    _load_labels,
    prepare_batches,
    prepare_mnist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_images_gz(tmp_path, name, n=4, pixels=4):
    """Create a minimal MNIST images .gz file."""
    h = w = int(pixels ** 0.5)
    buf = io.BytesIO()
    data = np.arange(n * pixels, dtype=np.uint8).reshape(n, pixels)
    header = struct.pack(">IIII", 2051, n, h, w)
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(header)
        gz.write(data.tobytes())
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


def _make_labels_gz(tmp_path, name, n=4):
    """Create a minimal MNIST labels .gz file."""
    buf = io.BytesIO()
    labels = np.arange(n, dtype=np.uint8) % 10
    header = struct.pack(">II", 2049, n)
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(header)
        gz.write(labels.tobytes())
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


# ---------------------------------------------------------------------------
# DATASETS registry
# ---------------------------------------------------------------------------


def test_datasets_registry():
    assert "mnist" in DATASETS


def test_prepare_dataset_unknown():
    with pytest.raises(ValueError, match="Unknown dataset"):
        prepare_dataset("definitely_not_a_dataset")


def test_prepare_dataset_calls_handler(tmp_path):
    fake_result = {"train_images": "a", "train_labels": "b",
                   "test_images": "c", "test_labels": "d"}
    with patch.dict(DATASETS, {"fake": lambda: fake_result}):
        result = prepare_dataset("fake")
    assert result == fake_result


# ---------------------------------------------------------------------------
# _get_cache_dir
# ---------------------------------------------------------------------------


def test_get_cache_dir_default():
    d = _get_cache_dir()
    assert isinstance(d, Path)


def test_get_cache_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MININNVERIFIER_CACHE_DIR", str(tmp_path / "cache"))
    d = _get_cache_dir()
    assert d == tmp_path / "cache"


# ---------------------------------------------------------------------------
# _load_images / _load_labels
# ---------------------------------------------------------------------------


def test_load_images(tmp_path):
    p = _make_images_gz(tmp_path, "imgs.gz", n=4, pixels=4)
    imgs = _load_images(p)
    assert imgs.shape == (4, 4)
    assert imgs.dtype == np.float64
    # Values should be in [0, 1]
    assert imgs.min() >= 0.0
    assert imgs.max() <= 1.0


def test_load_images_magic_error(tmp_path):
    buf = io.BytesIO()
    bad_header = struct.pack(">IIII", 9999, 4, 2, 2)
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(bad_header)
        gz.write(b"\x00" * 16)
    p = tmp_path / "bad_imgs.gz"
    p.write_bytes(buf.getvalue())
    with pytest.raises(AssertionError, match="Invalid magic"):
        _load_images(p)


def test_load_labels(tmp_path):
    p = _make_labels_gz(tmp_path, "lbls.gz", n=5)
    lbls = _load_labels(p, num_classes=10)
    assert lbls.shape == (5, 10)
    assert lbls.dtype == np.float64
    # One-hot: each row sums to 1
    row_sums = lbls.sum(axis=1)
    np.testing.assert_array_equal(row_sums, np.ones(5))


def test_load_labels_magic_error(tmp_path):
    buf = io.BytesIO()
    bad_header = struct.pack(">II", 8888, 3)
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(bad_header)
        gz.write(b"\x00" * 3)
    p = tmp_path / "bad_lbls.gz"
    p.write_bytes(buf.getvalue())
    with pytest.raises(AssertionError, match="Invalid magic"):
        _load_labels(p)


# ---------------------------------------------------------------------------
# _download_mnist
# ---------------------------------------------------------------------------


def test_download_mnist_already_present(tmp_path):
    """Files already present should not trigger download."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Create all expected files
    from testrunner.datasets.mnist import _FILES
    for f in _FILES:
        (raw_dir / f).write_bytes(b"fake")
    with patch("urllib.request.urlretrieve") as mock_dl:
        _download_mnist(raw_dir)
    mock_dl.assert_not_called()


def test_download_mnist_missing_files(tmp_path):
    """Missing files should be downloaded."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def fake_retrieve(url, dest):
        Path(dest).write_bytes(b"downloaded")

    with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
        _download_mnist(raw_dir)
    from testrunner.datasets.mnist import _FILES
    for f in _FILES:
        assert (raw_dir / f).exists()


# ---------------------------------------------------------------------------
# _convert_to_bin
# ---------------------------------------------------------------------------


def test_convert_to_bin(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"

    _make_images_gz(raw_dir, "train-images-idx3-ubyte.gz", n=4, pixels=4)
    _make_labels_gz(raw_dir, "train-labels-idx1-ubyte.gz", n=4)
    _make_images_gz(raw_dir, "t10k-images-idx3-ubyte.gz", n=2, pixels=4)
    _make_labels_gz(raw_dir, "t10k-labels-idx1-ubyte.gz", n=2)

    _convert_to_bin(raw_dir, out_dir)

    assert (out_dir / "train_images.bin").exists()
    assert (out_dir / "train_labels.bin").exists()
    assert (out_dir / "test_images.bin").exists()
    assert (out_dir / "test_labels.bin").exists()

    imgs = np.fromfile(out_dir / "train_images.bin", dtype=np.float64)
    assert imgs.shape == (4 * 4,)
    lbls = np.fromfile(out_dir / "train_labels.bin", dtype=np.float64)
    assert lbls.shape == (4 * 10,)


# ---------------------------------------------------------------------------
# prepare_batches
# ---------------------------------------------------------------------------


def test_prepare_batches_creates_files(tmp_path):
    n_samples = 8
    eval_batch_size = 4
    in_size = 3

    images = np.arange(n_samples * in_size, dtype=np.float64)
    images.tofile(tmp_path / "train_images.bin")

    batches = prepare_batches(tmp_path, "train", in_size, eval_batch_size)
    assert len(batches) == 2
    for p in batches:
        arr = np.fromfile(p, dtype=np.float64)
        assert arr.shape == (eval_batch_size * in_size,)


def test_prepare_batches_cached(tmp_path):
    """Second call should return cached files without recomputing."""
    n_samples = 4
    eval_batch_size = 2
    in_size = 2

    images = np.zeros(n_samples * in_size, dtype=np.float64)
    images.tofile(tmp_path / "test_images.bin")

    first = prepare_batches(tmp_path, "test", in_size, eval_batch_size)
    # Corrupt the source file — second call should still return cached
    (tmp_path / "test_images.bin").write_bytes(b"garbage")
    second = prepare_batches(tmp_path, "test", in_size, eval_batch_size)
    assert [p.name for p in first] == [p.name for p in second]


def test_prepare_batches_non_divisible(tmp_path):
    n_samples = 5
    eval_batch_size = 2
    in_size = 2

    images = np.zeros(n_samples * in_size, dtype=np.float64)
    images.tofile(tmp_path / "train_images.bin")

    with pytest.raises(ValueError, match="not divisible"):
        prepare_batches(tmp_path, "train", in_size, eval_batch_size)


# ---------------------------------------------------------------------------
# prepare_mnist (integration, with mocks)
# ---------------------------------------------------------------------------


def test_prepare_mnist_downloads_and_converts(tmp_path, monkeypatch):
    monkeypatch.setenv("MININNVERIFIER_CACHE_DIR", str(tmp_path))
    raw_dir = tmp_path / "mnist" / "raw"

    def fake_download(raw_dir_arg):
        raw_dir_arg = Path(raw_dir_arg)
        raw_dir_arg.mkdir(parents=True, exist_ok=True)
        # Create fake .gz files
        _make_images_gz(raw_dir_arg, "train-images-idx3-ubyte.gz", n=4, pixels=4)
        _make_labels_gz(raw_dir_arg, "train-labels-idx1-ubyte.gz", n=4)
        _make_images_gz(raw_dir_arg, "t10k-images-idx3-ubyte.gz", n=2, pixels=4)
        _make_labels_gz(raw_dir_arg, "t10k-labels-idx1-ubyte.gz", n=2)

    with patch("testrunner.datasets.mnist._download_mnist", side_effect=fake_download):
        result = prepare_mnist()

    assert "train_images" in result
    assert "test_labels" in result
    for v in result.values():
        assert Path(v).exists()


def test_prepare_mnist_uses_cache(tmp_path, monkeypatch):
    """If .bin files already exist, _convert_to_bin should not be called."""
    monkeypatch.setenv("MININNVERIFIER_CACHE_DIR", str(tmp_path))
    cache_dir = tmp_path / "mnist"
    cache_dir.mkdir(parents=True)
    # Pre-create all output .bin files
    for name in ["train_images.bin", "train_labels.bin", "test_images.bin", "test_labels.bin"]:
        (cache_dir / name).write_bytes(b"cached")

    with patch("testrunner.datasets.mnist._download_mnist") as mock_dl, \
         patch("testrunner.datasets.mnist._convert_to_bin") as mock_conv:
        result = prepare_mnist()

    mock_conv.assert_not_called()
