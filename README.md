# Mini NN Verifier
A Rust implementation of a neural network verifier based on the original educational
implementation [mininnverifier (The Mini Neural Network Verifier)](https://github.com/cherrywoods/mininnverifier)
by [cherrywoods](https://github.com/cherrywoods) and compatible with the
[mininnverifier testrunner](https://github.com/cherrywoods/mininnverifier-testrunner).

## Status

| Milestone | Scope | Status |
|-----------|-------|--------|
| 1 | Eval, grad, MLP training | Finished (for now) |
| 2 | Interval bounds, verify | In Progress |
| 3 | Affine bounds, verify2 | Not started |

## Architecture

```
src/
├── cli/
│   ├── eval.rs       # `mininn eval` subcommand
│   ├── grad.rs       # `mininn grad` subcommand
│   └── train.rs      # `mininn train` subcommand
├── interpreters/
│   ├── eval.rs       # Forward-pass interpreter (EvalInterpreter)
│   ├── grad.rs       # Backward-mode autodiff (GradInterpreter)
│   └── eval_util.rs  # Tensor ops and broadcasting helpers
└── mininn/
    ├── nn.rs         # ComputeGraph, Primitive, Atom types
    ├── parse.rs      # .mininn ZIP parser
    ├── mlp.rs        # MLP initialisation (Kaiming uniform) and graph builder
    └── env.rs        # Variable environment (name → ArrayD<f64>)
```

## CLI

```
mininn eval  --output-dir <dir> <network.mininn> <input.bin> [...]
mininn grad  --output-dir <dir> <network.mininn> <input.bin> [...]
mininn train --output-dir <dir> <dataset>        <inputs.bin> <labels.bin>
```

Input and output arrays are flat row-major `float64` binary files (no header).
Network files are ZIP archives containing `graph.txt` and one `.bin` file per
named constant.

## Training

Training uses Adam (β₁=0.9, β₂=0.999, ε=1e-8) with softmax cross-entropy loss.
Hyperparameters are loaded from `hyperparams/<dataset>.json` relative to the
working directory. Checkpoints are saved as `.mininn` files after each epoch.

Example hyperparameter file (`hyperparams/mnist.json`):
```json
{
    "in_size": 784,
    "layer_sizes": [128, 10],
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "eval_batch_size": 10000,
    "rng_key": 0
}
```

## Building

```bash
cargo build --release
```

## Running tests

Tests use the Python testrunner. The recommended way is via Docker so the
working directory and file paths are handled correctly:

```bash
# Build the image
docker build -t mininn:local .

# Run a milestone's tests
python -m testrunner docker mininn:local tests/milestone1/base/unit

# Or run locally from the repo root (hyperparams/ must be in cwd)
tests/testrunner/.venv/bin/python -m testrunner local \
    "$(pwd)/target/release/mininn" tests/milestone1/base/unit
```

The testrunner requires Python >= 3.14. Install it with:

```bash
pip install uv
uv pip install tests/testrunner/
```

For more details refer to the submodule.

## CI

GitHub Actions runs milestone tests on every push. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). Train tests are
excluded from the regular pipeline (slow, requires MNIST download) and run only
on manual dispatch.
