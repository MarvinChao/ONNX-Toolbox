"""
Regression test driver for the ONNX-Toolbox.

Builds a representative, hermetic multi-op ONNX model, runs it through the
toolbox analysis pipeline, and compares the per-node stats against a golden
baseline stored under utils/golden/.

Usage:
    # Compare current output against the golden baseline (fails on drift)
    python utils/run_regression.py

    # Regenerate the golden baseline after an intentional change
    python utils/run_regression.py --update

    # Also run the per-handler unit checks
    python utils/run_regression.py --with-handlers
"""

import argparse
import os
import sys

import numpy as np
from onnx import TensorProto

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.model_builder import (  # noqa: E402
    make_initializer,
    make_sequential_model,
    make_tensor_value_info,
)
from utils.regression import RegressionRunner, analyze_model  # noqa: E402

GOLDEN_DIR = os.path.join(_REPO_ROOT, "utils", "golden")
GOLDEN_CASE = "reference_cnn"


def build_reference_model():
    """
    A small CNN-ish graph exercising a spread of handlers:
    Conv -> Clip -> AveragePool -> Reshape -> Gemm.
    """
    x = make_tensor_value_info("input", [1, 3, 32, 32])

    conv_w = make_initializer("conv_w", np.ones((8, 3, 3, 3)))
    clip_min = make_initializer("clip_min", 0.0)
    clip_max = make_initializer("clip_max", 6.0)
    reshape_shape = make_initializer(
        "reshape_shape", [1, 8 * 15 * 15], elem_type=TensorProto.INT64
    )
    gemm_b = make_initializer("gemm_b", np.ones((8 * 15 * 15, 10)))
    gemm_c = make_initializer("gemm_c", np.ones((10,)))

    specs = [
        {
            "op_type": "Conv",
            "output_info": make_tensor_value_info("conv_out", [1, 8, 30, 30]),
            "initializers": [conv_w],
            "extra_inputs": ["conv_w"],
            "attributes": {"kernel_shape": [3, 3], "group": 1},
        },
        {
            "op_type": "Clip",
            "output_info": make_tensor_value_info("clip_out", [1, 8, 30, 30]),
            "initializers": [clip_min, clip_max],
            "extra_inputs": ["clip_min", "clip_max"],
        },
        {
            "op_type": "AveragePool",
            "output_info": make_tensor_value_info("pool_out", [1, 8, 15, 15]),
            "attributes": {"kernel_shape": [2, 2], "strides": [2, 2]},
        },
        {
            "op_type": "Reshape",
            "output_info": make_tensor_value_info("reshape_out", [1, 8 * 15 * 15]),
            "initializers": [reshape_shape],
            "extra_inputs": ["reshape_shape"],
        },
        {
            "op_type": "Gemm",
            "output_info": make_tensor_value_info("output", [1, 10]),
            "initializers": [gemm_b, gemm_c],
            "extra_inputs": ["gemm_b", "gemm_c"],
        },
    ]
    return make_sequential_model(specs, x, "output")


def run(update=False, with_handlers=False):
    exit_code = 0

    if with_handlers:
        from utils import test_handlers

        print("=== handler unit checks ===")
        rc = test_handlers.main()
        exit_code |= rc
        print()

    print("=== golden regression ===")
    model = build_reference_model()
    stats = analyze_model(model, run_shape_inference=False, check=True)
    runner = RegressionRunner(GOLDEN_DIR)

    if update:
        runner.save(GOLDEN_CASE, stats)
        print(f"Updated golden baseline: {os.path.join(GOLDEN_DIR, GOLDEN_CASE)}.json")
        return exit_code

    diffs = runner.compare(GOLDEN_CASE, stats)
    if diffs:
        print(f"REGRESSION DETECTED ({len(diffs)} difference(s)):")
        for d in diffs:
            print(f"  - {d}")
        exit_code |= 1
    else:
        print(f"PASS: {len(stats)} nodes match golden baseline '{GOLDEN_CASE}'.")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="ONNX-Toolbox regression runner")
    parser.add_argument(
        "--update", action="store_true", help="Regenerate the golden baseline"
    )
    parser.add_argument(
        "--with-handlers", action="store_true", help="Also run per-handler unit checks"
    )
    args = parser.parse_args()
    return run(update=args.update, with_handlers=args.with_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
