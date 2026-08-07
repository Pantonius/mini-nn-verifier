# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Shared fixtures for testrunner unit tests."""

import gzip
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# File / directory helpers
# ---------------------------------------------------------------------------


def make_test_dir(tmp_path, config, files=None):
    """Create a test directory with test.json and optional extra files.

    files: dict mapping relative path string -> content (bytes or ndarray).
    """
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test.json").write_text(json.dumps(config))
    if files:
        for rel, content in files.items():
            dest = test_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, np.ndarray):
                content.tofile(dest)
            elif isinstance(content, bytes):
                dest.write_bytes(content)
            else:
                dest.write_text(content)
    return test_dir


def make_float64_bin(tmp_path, name, data):
    """Write a float64 numpy array to a .bin file and return its Path."""
    p = tmp_path / name
    np.array(data, dtype=np.float64).tofile(p)
    return p


def make_mininn_zip(tmp_path, name, graph_txt, constants=None):
    """Create a .mininn (zip) file and return its Path."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        with zf.open("graph.txt", "w") as f:
            f.write(graph_txt.encode())
        if constants:
            for cname, arr in constants.items():
                with zf.open(f"{cname}.bin", "w") as f:
                    f.write(np.array(arr, dtype=np.float64).tobytes())
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


def make_mnist_gz(tmp_path, kind, data):
    """Create a minimal MNIST-format .gz file.

    kind: 'images' or 'labels'
    data: for images, shape (N, H*W) uint8; for labels, shape (N,) uint8
    """
    buf = io.BytesIO()
    data = np.array(data, dtype=np.uint8)
    if kind == "images":
        n, pixels = data.shape
        h = w = int(pixels ** 0.5)
        header = struct.pack(">IIII", 2051, n, h, w)
    else:
        n = data.shape[0]
        header = struct.pack(">II", 2049, n)
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(header)
        gz.write(data.tobytes())
    p = tmp_path / f"mnist_{kind}.gz"
    p.write_bytes(buf.getvalue())
    return p


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_eval_config():
    return {
        "command": "eval",
        "network": "net.mininn",
        "inputs": ["input.bin"],
        "expected_outputs": ["expected_output.bin"],
    }


@pytest.fixture
def simple_grad_config():
    return {
        "command": "grad",
        "network": "net.mininn",
        "inputs": ["input.bin"],
        "expected_outputs": ["expected_grad.bin"],
    }
