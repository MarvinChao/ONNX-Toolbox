# Regression Test Tooling

The ONNX-Toolbox ships a hermetic (no-download, fully deterministic) regression
test suite. Its job is to lock down operator handler behaviour so that any
future change which alters the analysis results is caught early.

> **Where the code lives:** all of the tooling described in this document is in
> the [`utils/`](../utils) directory at the repository root. This document lives
> in `docs/` for reference only — there is no test code in `docs/`. All file
> paths below (e.g. `utils/model_builder.py`) and all commands are relative to
> the repository root.

Everything in `utils/` uses only `onnx` and `numpy` (already in
`requirements.txt`). No models are downloaded and no randomness is used, so
results are reproducible on any machine.

---

## Why this exists

The toolbox analyzes an ONNX model by dispatching each graph node to an
operator handler (in `handlers/`), which computes compute-primitive counts
(MAC / ALU / EXP / DIV / TRIG / SQRT), data sizes, and captured attributes.

Small changes to a handler (or to the shared `NodeAttributes` structure) can
silently change those numbers. These tests pin the expected output so a
regression shows up as an explicit, readable diff instead of going unnoticed.

---

## Layout

All of the following files are in the `utils/` directory:

| File | Purpose |
|------|---------|
| `utils/model_builder.py` | Build small, deterministic ONNX graphs (single-op or a linear multi-op chain) with explicit tensor shapes. |
| `utils/regression.py` | Run a model through the real handler pipeline and save / compare per-node stats against a golden baseline. |
| `utils/test_handlers.py` | Per-handler checks asserting compute counts and captured attributes for each supported op. |
| `utils/run_regression.py` | Command-line driver: builds a reference model, compares to the golden baseline, and can regenerate it. |
| `utils/golden/reference_cnn.json` | The checked-in golden baseline for the reference model. |

---

## How it works

### 1. Building test models (`utils/model_builder.py`)

Handlers read tensor shapes directly from the graph (`value_info`, `input`,
`output`), so the builders attach explicit shapes to every tensor. This means
tests do not depend on shape inference and stay deterministic.

Key helpers:

- `make_tensor_value_info(name, shape, elem_type)` — a typed, shaped tensor.
- `make_initializer(name, array, elem_type)` — a constant tensor (weights,
  `shape` for Reshape, `pads` for Pad, etc.). Handlers treat initializer inputs
  as model coefficients and add them to `weight_size`.
- `make_single_node_model(...)` — a valid one-node model, used by the
  per-handler checks.
- `make_sequential_model(...)` — a linear chain of nodes where each node's
  input is the previous node's output. Used for the end-to-end golden test.

### 2. Running the pipeline (`utils/regression.py`)

`analyze_model(model, run_shape_inference=False, check=False)` mirrors what
`onnx_analysis.ModelStats.parse_model` does:

1. optionally run `onnx.checker`,
2. optionally run shape inference (ORT symbolic, falling back to onnx),
3. dispatch every node via `get_handler(op_type).handle(model, node)`,
4. collect each node's `to_dict()`.

Importing `handlers` (done inside the module) registers every handler in the
`node_registry` as a side effect.

`RegressionRunner(golden_dir)` handles the baseline:

- `save(name, stats)` — write the golden JSON (keys sorted, numpy values
  normalized to plain Python for stable diffs).
- `load(name)` — read a golden JSON, or `None` if missing.
- `compare(name, stats)` — return a list of human-readable differences; an
  empty list means the current output matches the baseline.

### 3. Per-handler checks (`utils/test_handlers.py`)

Each `test_*` function builds a single-op graph and asserts the expected
compute counts and captured attributes. Coverage includes:

- Compute ops: `Conv`, `Gemm`, `AveragePool`, `Clip`, `BatchNormalization`,
  `InstanceNormalization`, `Mul`.
- Data-movement / shape ops: `Reshape`, `Div`, `Gather`, `Slice`, `Pad`.
- Regression edge cases: `Conv` with an unresolved data shape (channels derived
  from the weight tensor), `Gemm` with a 1-D input, and `Mul` with broadcasting
  operands of different rank.

### 4. End-to-end golden test (`utils/run_regression.py`)

`build_reference_model()` assembles a small CNN-like chain:

```
Conv -> Clip -> AveragePool -> Reshape -> Gemm
```

The driver runs it through `analyze_model` and compares the resulting per-node
stats against `utils/golden/reference_cnn.json`. If any number or attribute
changes, the comparison prints exactly which node, which field, and the old vs
new value.

---

## Usage

Run these from the repository root (not from inside `docs/` or `utils/`).

```bash
# Compare the reference model against the golden baseline (exits non-zero on drift)
python utils/run_regression.py

# Regenerate the golden baseline after an intentional, reviewed change
python utils/run_regression.py --update

# Also run the per-handler unit checks
python utils/run_regression.py --with-handlers
```

The per-handler checks can also be run on their own, either standalone or with
pytest:

```bash
# Standalone: prints PASS/FAIL per check and a summary
python utils/test_handlers.py

# With pytest (if installed)
pytest utils/test_handlers.py
```

---

## Typical workflows

**Everyday check — did I break anything?**

```bash
python utils/run_regression.py --with-handlers
```

A clean run means handler outputs are unchanged. Any failure lists the exact
handler and field that drifted.

**I intentionally changed a handler's math or added an attribute.**

1. Update / add the relevant assertion in `utils/test_handlers.py` so the
   expected values reflect the new behaviour.
2. Regenerate the golden baseline:
   ```bash
   python utils/run_regression.py --update
   ```
3. Review the diff of `utils/golden/reference_cnn.json` to confirm the change is
   what you expect, then commit both the code and the updated baseline together.

**I added a brand-new operator handler.**

1. Add a `test_<op>` function in `utils/test_handlers.py` that builds a
   single-node graph and asserts the expected counts / attributes.
2. Optionally add the op to the reference chain in `utils/run_regression.py` and
   regenerate the golden baseline so it is exercised end-to-end.

---

## Extending the suite

- **New single-op check:** add a `test_<op>()` in `utils/test_handlers.py` using
  `make_single_node_model` (via the local `_run` helper) and assert on the
  returned `NodeAttributes`.
- **New golden model:** build it in `utils/run_regression.py` (or a new driver)
  with `make_sequential_model`, give it a case name, and call
  `RegressionRunner.save(name, stats)` once to create its baseline JSON under
  `utils/golden/`.
- **Broadcasting / multi-input ops:** remember that `input_dimension` is a list
  of per-input shapes. The data tensor is `input_dimension[0]`; element-wise
  counts should use `output_dimension` to account for broadcasting.

---

## Notes and limitations

- The suite is intentionally hermetic: it does not download or require external
  model files, which keeps it fast and CI-friendly.
- It validates the analysis numbers and attribute capture, not numerical
  inference correctness of the models themselves.
- `utils/golden/reference_cnn.json` is a committed artifact. Treat changes to it
  the same as code changes and review them before committing.
- Generated `__pycache__` directories are already ignored via `.gitignore`.
