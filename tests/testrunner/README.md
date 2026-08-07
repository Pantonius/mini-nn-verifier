# mininnverifier Test Runner

A test runner for neural network verifier implementations. It discovers test
cases from directories, runs an implementation against them, checks outputs,
and reports scores.

## Overview

Tests live in directories that each contain a `test.json` file describing what
to run and what to check. The runner supports three backends:

- **docker** — runs the implementation inside a Docker container (the test
  directory is mounted at `/data`)
- **podman** — same as docker, but uses `podman run` (rootless-friendly);
  identical mount layout and path rewriting
- **local** — runs a local command directly

Twelve test modes are supported, grouped by what they exercise:

- **functional** — `eval`, `grad`, `bounds`, `verify`
- **training** — `train`
- **fuzzing** — `fuzz_eval`, `fuzz_grad`, `fuzz_bounds`
- **benchmarking** — `bench_eval`, `bench_grad`, `bench_bounds`, `bench_verify`

The runner discovers all tests under a root directory, sorts them
cheapest-first, and writes the progress to the terminal.

## Commands

### Run tests

```
python -m testrunner {docker,podman,local} backend_arg test_dir [--generate] \
    [--output {cli,json}] [--extra-run-args "..."]
```

Discovers every `test.json` beneath `test_dir`, runs them, and prints a
pass/fail summary. Exits with code 1 if any test fails.

```
python -m testrunner docker my-image:latest tests/milestone1
python -m testrunner podman my-image:latest tests/milestone1
python -m testrunner local "python -m myimpl" tests/milestone1
```

`--extra-run-args` passes additional flags through to `docker run` / `podman
run` (shell-quoted; ignored for the `local` backend). Use it to apply
sandboxing or resource limits, e.g.

```
python -m testrunner podman my-image:latest tests/milestone1 \
    --extra-run-args "--network=none --memory=1g --pids-limit=256 --read-only \
                      --tmpfs=/tmp --cap-drop=ALL --security-opt=no-new-privileges"
```

For details:

```
python -m testrunner -h
```

### List tests

```
python -m testrunner.list test_dir [--output {cli,json}]
```

Prints all tests grouped by directory, showing the test name and command type.

```
python -m testrunner.list tests/milestone1
```

```
python -m testrunner.list -h
```

### Inspect a test

```
python -m testrunner.show test_dir
```

Shows the network graph, input arrays, expected outputs, and (after a run) the
actual outputs for every test found under `test_dir`. For fuzz tests it prints
each saved failing case with its error, implementation output, network graph,
and inputs.

```
python -m testrunner.show tests/milestone1/base/unit/eval/add
python -m testrunner.show tests/milestone1   # show all
```

```
python -m testrunner.show -h
```

### Reproduce a fuzz failure

```
python -m testrunner.reproduce {docker,podman,local} backend_arg failure_dir
```

Reruns the exact network and inputs from a saved fuzz failure (a directory
created automatically inside `actual/fuzz_failures/` when a fuzz test fails).
Prints the network, inputs, and outcome.

```
python -m testrunner.reproduce local "python -m myimpl" \
    tests/milestone1/base/fuzz/eval/actual/fuzz_failures/0
```

```
python -m testrunner.reproduce -h
```

## Implementation interface

The runner invokes the implementation as a subprocess. This section documents
the CLI contract and file formats the implementation must follow.

### CLI

These commands share a common argument structure:

```
<impl> eval   --output-dir <dir> <network.mininn> <input1.bin> [<input2.bin> ...]
<impl> grad   --output-dir <dir> <network.mininn> <input1.bin> [<input2.bin> ...]
<impl> bounds --output-dir <dir> <network.mininn> box <lb.bin> <ub.bin>
<impl> verify --output-dir <dir> <network.mininn> box <lb.bin> <ub.bin>
<impl> train  --output-dir <dir> <dataset-name>  <train_inputs.bin> <train_labels.bin>
```

`<impl>` is the `backend_arg` — either a container image name (docker/podman)
or a shell command (e.g. `python -m myimpl`). For the container backends
(docker, podman) the test directory is mounted at `/data` and all paths are
rewritten accordingly.

For `bounds` and `verify` the inputs are a flat list interleaving inline
markers (`box`, `point`) with file paths: `box <lb.bin> <ub.bin>` denotes an
axis-aligned input box, `point <x.bin>` a concrete point. Markers are passed
through verbatim; only file paths are rewritten for the backend.

**`eval` and `grad`** must:

- Write each output array to a file inside `--output-dir`
- Print the absolute path of each output file to stdout, one per line
- Exit 0 on success, non-zero on failure

**`bounds`** must:

- Compute a lower and upper bound for each network output over the input box
- Write `output_<i>_lb.bin` and `output_<i>_ub.bin` per output inside
  `--output-dir`, and print their paths to stdout — lower bound before upper
  bound for each output
- Exit 0 on success, non-zero on failure

**`verify`** must:

- Decide whether the network's scalar output (a *margin*) satisfies
  `margin(x) >= 0` for all `x` in the input box
- Print the verdict as the **last** stdout line: `sat` (holds) or `viol`
  (violated); any logging may precede it
- For a `viol` verdict, write one `counterexample_<i>.bin` per network input
  inside `--output-dir` and print their path(s) on the line(s) immediately
  before the verdict
- Exit 0 on success, non-zero on failure

**`train`** must:

- Print `eval_batch_size: <N>` as the first output line
- Print the absolute path of each saved checkpoint file, one per line
- Checkpoints must be written inside `--output-dir` (for docker compatibility)
- Exit 0 on success, non-zero on failure

### `.mininn` file format

A `.mininn` file is a ZIP archive containing:

- **`graph.txt`** — the compute graph in a text format (see below)
- **`<NAME>.bin`** — one file per graph constant, in the same flat float64
  format as input/output `.bin` files (see below)

#### `graph.txt` syntax

```
input: <var>; <var>; ...
<out_var> = <primitive>{<option>: <value>; ...} <in_var> [<in_var> ...]
...
output: <var>; <var>; ...
```

Each variable is written as `name[d0,d1,...]` where `d0,d1,...` are the
dimension sizes. Options are omitted when empty (`{}`).

Example:

```
input: a[2,3]; b[2,3]
c[2,3] = add{} a b
d[2,3] = relu{} c
e[1,2,3] = expand_dims{'axes': (0,)} d
output: e[1,2,3]
```

Constants (uppercase names such as `A`, `B`) are referenced in equations like
any other variable; their values are loaded from the correspondingly named
`.bin` file inside the ZIP.

### `.bin` file format

All array data — inputs, outputs, gradients, and graph constants — are stored
as flat, row-major (C-order) `float64` arrays with no header. The shape is
conveyed through the graph description; the `.bin` file only contains
the raw `8 * n` bytes.

To write such a file in Python:

```python
import numpy as np
array.astype(np.float64).tofile(path)   # C-order by default
```

To read it back in Python:

```python
data = np.fromfile(path, dtype=np.float64)   # then reshape as needed
```

Writing a `.bin` file in C++:

```cpp
void write_bin(const std::string& path, const double* data, std::size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), n * sizeof(double));
}
```

To read it back in C++:

```cpp
std::vector<double> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::size_t n_bytes = f.tellg();
    f.seekg(0);
    std::vector<double> data(n_bytes / sizeof(double));
    f.read(reinterpret_cast<char*>(data.data()), n_bytes);
    return data;   // flat, row-major; reshape using known shape
}
```

## Test modes

### `eval` — forward pass

Runs the implementation's `eval` command on a fixed network and input(s).
Compares the output `.bin` files to expected values within a configurable
absolute tolerance (default `1e-4`).

```json
{
  "command": "eval",
  "network": "network.mininn",
  "inputs": ["input.bin"],
  "expected_outputs": ["expected_output.bin"],
  "tolerance": 1e-4
}
```

Default timeout: 60 s. Default scoring: **binary** (full points if pass, 0 otherwise).

### `grad` — gradient computation

Identical structure to `eval` but calls the `grad` command. The output
is a gradient array per input, checked against expected values with the same
tolerance mechanism.

```json
{
  "command": "grad",
  "network": "network.mininn",
  "inputs": ["input.bin"],
  "expected_outputs": ["expected_grad.bin"],
  "tolerance": 1e-4
}
```

Default timeout: 60 s. Default scoring: **binary**.

### `bounds` — interval output bounds

Runs the `bounds` command on a network and an input box, producing a lower and
upper bound `.bin` file per network output. The `bounds_within_range` check
validates them for:

- **Soundness** — every pre-evaluated `sample_outputs` value (at points inside
  the box) lies within the impl's bounds, with absolute slack `sample_atol`.
- **Tightness** (optional) — if reference IBP bounds are supplied, the impl's
  box width may not exceed the reference width by more than
  `(tightness_factor - 1)` on either side.

```json
{
  "command": "bounds",
  "network": "network.mininn",
  "inputs": ["box", "input_0_lb.bin", "input_0_ub.bin"],
  "check": "bounds_within_range",
  "output_shapes": [[10]],
  "sample_outputs": ["sample_outputs_0.bin"],
  "sample_atol": 1e-9,
  "reference_lb": ["reference_output_0_lb.bin"],
  "reference_ub": ["reference_output_0_ub.bin"],
  "tightness_factor": 1.1
}
```

`output_shapes` and `sample_outputs` are required; `reference_lb`/`reference_ub`
and `tightness_factor` (default `1.5`) enable the optional tightness check.

Default timeout: 60 s. Default scoring: **binary**.

### `verify` — robustness verification

Runs the `verify` command, which decides whether a network's scalar margin
output satisfies `margin(x) >= 0` for all `x` in the input box, and emits a
`sat`/`viol` verdict. The runner checks the verdict against `expected_verdict`;
for a `viol` verdict it re-evaluates the reported counterexample (via `eval`)
and confirms the witness is inside the box and has `margin < 0` — any valid
witness is accepted, not just a stored one.

```json
{
  "command": "verify",
  "network": "network.mininn",
  "inputs": ["box", "input_0_lb.bin", "input_0_ub.bin"],
  "expected_verdict": "sat",
  "box_atol": 1e-9
}
```

Default timeout: 300 s. Default scoring: **binary**.

### `train` — training with accuracy evaluation

Runs the `train` command, which must print `eval_batch_size: N` on its first
output line followed by checkpoint file paths (one per line). The runner then
calls `eval` on each checkpoint to measure train and test accuracy, reporting
the best test accuracy achieved.

Dataset files can be specified explicitly or loaded from a named cache
(e.g. `"source_dataset": "mnist"`).

```json
{
  "command": "train",
  "dataset": "mnist_mlp",
  "in_size": 784,
  "num_classes": 10,
  "source_dataset": "mnist"
}
```

Checkpoints **must** be saved inside the directory passed via `--output-dir`
(when using the docker backend, only that directory is mounted).

Default timeout: 600 s. Default scoring: **exponential** based on best test
accuracy (higher k rewards improvements near 100% more heavily).

### `fuzz_eval` / `fuzz_grad` / `fuzz_bounds` — robustness fuzzing

Uses [Hypothesis](https://hypothesis.readthedocs.io/) to generate random
compute graphs and matching inputs, then calls `eval`, `grad`, or `bounds` on
each one. A trial fails if the command crashes, times out, produces the wrong
number of output files, outputs arrays of the wrong shape, or (optionally)
outputs NaN/Inf values.

Failing cases are saved to `actual/fuzz_failures/<n>/` and can be inspected
with `python -m testrunner.show` or reproduced with
`python -m testrunner.reproduce`.

```json
{
  "command": "fuzz_eval",
  "n_trials": 500,
  "seed": 0,
  "primitives": "all",
  "check_nan_inf": false
}
```

`primitives` can be `"all"` (default), `"safe"` (excludes `log`, `sqrt`,
`reciprocal`, `exp`), or an explicit list of primitive names.

`fuzz_bounds` additionally widens each random input into an axis-aligned box and
checks the returned bounds for **soundness** (random points inside the box stay
within the bounds) and optional finiteness. The box half-width `eps` is drawn
per trial from `[eps_min, eps_max]` (default `[0.0, 0.5]`; set `eps` as a
shorthand for `eps_max`). Soundness sampling is tunable via `n_eval_samples`
(default 3), `sample_atol`, and `sample_rtol` (default `1e-6`).

```json
{
  "command": "fuzz_bounds",
  "n_trials": 500,
  "seed": 0,
  "primitives": "all",
  "eps_min": 0.0,
  "eps_max": 0.5,
  "check_nan_inf": false
}
```

Default timeout: 600 s. Default scoring: **proportional** to fraction of
trials passed.

### `bench_eval` / `bench_grad` / `bench_bounds` / `bench_verify` — performance benchmarking

Runs the underlying command (`eval`, `grad`, `bounds`, or `verify`) repeatedly
(with warm-up) and measures median wall-clock time. The test passes when the
SUT median does not exceed the reference median by more than `max_slowdown`.
The `inputs` follow the same convention as the underlying mode (plain `.bin`
paths for `eval`/`grad`; `box`/`point` markers for `bounds`/`verify`).

Reference times are generated with `--generate` and stored in
`reference_time.json` inside the test directory.

```json
{
  "command": "bench_eval",
  "network": "network.mininn",
  "inputs": ["input.bin"],
  "n_repeats": 50,
  "max_slowdown": 2.0
}
```

Default timeout: 600 s (`bench_verify`: 900 s). Default scoring: **speed**.

## Examples

```bash
# 1. List available tests
python -m testrunner.list tests/milestone1

# 2. Generate expected outputs for eval/grad/bench tests (run once)
python -m testrunner local "python -m myimpl" tests/milestone1 --generate

# 3. Run the full test suite
python -m testrunner local "python -m myimpl" tests/milestone1

# 4. Inspect a failing test
python -m testrunner.show tests/milestone1/eval/hard

# 5. Inspect fuzz failures
python -m testrunner.show tests/milestone1/fuzz/eval

# 6. Reproduce a specific fuzz failure
python -m testrunner.reproduce local "python -m myimpl" \
    tests/milestone1/fuzz/eval/actual/fuzz_failures/0
```
