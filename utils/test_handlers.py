"""
Regression checks for individual operator handlers.

Each check builds a small synthetic ONNX graph for one op, dispatches it
through the real handler, and asserts the compute-primitive counts and
captured attributes. These lock down the handler behaviour so future changes
that alter analysis results are caught.

Runnable two ways:
    * pytest utils/test_handlers.py
    * python utils/test_handlers.py        (standalone, prints PASS/FAIL)
"""

import os
import sys

import numpy as np
from onnx import TensorProto

# Allow running as a standalone script from anywhere by ensuring the repo root
# (the parent of this utils/ directory) is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import handlers  # noqa: F401,E402  (populate registry)
from node_registry import get_handler  # noqa: E402
from utils.model_builder import (  # noqa: E402
    make_initializer,
    make_single_node_model,
    make_tensor_value_info,
)


def _run(op_type, inputs, outputs, input_infos, output_infos,
         initializers=None, value_infos=None, attributes=None, opset=13):
    model, node = make_single_node_model(
        op_type, inputs, outputs, input_infos, output_infos,
        initializers=initializers, value_infos=value_infos,
        attributes=attributes, opset=opset,
    )
    return get_handler(op_type).handle(model, node)


# --------------------------------------------------------------------------- #
# Compute-op handlers
# --------------------------------------------------------------------------- #

def test_conv():
    x = make_tensor_value_info("x", [1, 3, 32, 32])
    y = make_tensor_value_info("y", [1, 8, 30, 30])
    w = make_initializer("w", np.ones((8, 3, 3, 3)))
    a = _run(
        "Conv", ["x", "w"], ["y"], [x], [y],
        initializers=[w],
        attributes={"kernel_shape": [3, 3], "group": 1},
    )
    out = 1 * 8 * 30 * 30
    assert a.count_mac == out * 9 * 3, a.count_mac
    assert a.count_alu == out, a.count_alu


def test_conv_missing_data_shape_falls_back_to_weight():
    # Data input shape is unresolved (empty); channels must come from W.
    x = make_tensor_value_info("x", [])
    y = make_tensor_value_info("y", [1, 8, 30, 30])
    w = make_initializer("w", np.ones((8, 3, 3, 3)))
    a = _run(
        "Conv", ["x", "w"], ["y"], [x], [y],
        initializers=[w],
        attributes={"kernel_shape": [3, 3], "group": 1},
    )
    out = 1 * 8 * 30 * 30
    # input channels derived from weight_shape[1] * group = 3
    assert a.count_mac == out * 9 * 3, a.count_mac


def test_gemm():
    x = make_tensor_value_info("x", [4, 16])
    y = make_tensor_value_info("y", [4, 10])
    b = make_initializer("b", np.ones((16, 10)))
    c = make_initializer("c", np.ones((10,)))
    a = _run("Gemm", ["x", "b", "c"], ["y"], [x], [y], initializers=[b, c])
    assert a.count_mac == 4 * 16 * 10, a.count_mac
    assert a.count_alu == 4 * 10, a.count_alu


def test_gemm_1d_input():
    # 1-D A input (e.g. after a flatten/reshape) must not raise.
    x = make_tensor_value_info("x", [1280])
    y = make_tensor_value_info("y", [1000])
    b = make_initializer("b", np.ones((1280, 1000)))
    c = make_initializer("c", np.ones((1000,)))
    a = _run("Gemm", ["x", "b", "c"], ["y"], [x], [y], initializers=[b, c])
    assert a.count_mac == 1 * 1280 * 1000, a.count_mac


def test_averagepool():
    x = make_tensor_value_info("x", [1, 3, 32, 32])
    y = make_tensor_value_info("y", [1, 3, 16, 16])
    a = _run(
        "AveragePool", ["x"], ["y"], [x], [y],
        attributes={"kernel_shape": [2, 2], "strides": [2, 2]},
    )
    out = 1 * 3 * 16 * 16
    assert a.count_alu == out * (4 - 1), a.count_alu
    assert a.count_div == out, a.count_div
    assert a.kernel_shape == [2, 2]


def test_clip_relu6():
    x = make_tensor_value_info("x", [1, 16, 8, 8])
    y = make_tensor_value_info("y", [1, 16, 8, 8])
    cmin = make_initializer("min", 0.0)
    cmax = make_initializer("max", 6.0)
    a = _run("Clip", ["x", "min", "max"], ["y"], [x], [y], initializers=[cmin, cmax])
    n = 1 * 16 * 8 * 8
    assert a.count_alu == n * 2, a.count_alu
    assert a.clip_min == 0.0 and a.clip_max == 6.0


def test_batchnorm():
    x = make_tensor_value_info("x", [1, 64, 56, 56])
    y = make_tensor_value_info("y", [1, 64, 56, 56])
    inits = [make_initializer(n, np.ones((64,))) for n in ("s", "b", "m", "v")]
    a = _run(
        "BatchNormalization",
        ["x", "s", "b", "m", "v"], ["y"], [x], [y], initializers=inits,
    )
    n = 1 * 64 * 56 * 56
    assert a.count_mac == n, a.count_mac
    assert a.count_alu == n * 6, a.count_alu
    assert a.count_sqrt == 64, a.count_sqrt


def test_instancenorm():
    x = make_tensor_value_info("x", [2, 32, 24, 24])
    y = make_tensor_value_info("y", [2, 32, 24, 24])
    inits = [make_initializer(n, np.ones((32,))) for n in ("scale", "bias")]
    a = _run(
        "InstanceNormalization",
        ["x", "scale", "bias"], ["y"], [x], [y], initializers=inits,
    )
    n = 2 * 32 * 24 * 24
    assert a.count_mac == n, a.count_mac
    assert a.count_sqrt == 2 * 32, a.count_sqrt


def test_mul_broadcast():
    # Broadcasting operands with different ranks must not raise.
    x = make_tensor_value_info("x", [1, 3, 8, 8])
    s = make_tensor_value_info("s", [3, 1, 1])
    y = make_tensor_value_info("y", [1, 3, 8, 8])
    a = _run("Mul", ["x", "s"], ["y"], [x, s], [y])
    assert a.count_alu == 1 * 3 * 8 * 8, a.count_alu


# --------------------------------------------------------------------------- #
# Data-movement / shape handlers
# --------------------------------------------------------------------------- #

def test_reshape():
    x = make_tensor_value_info("x", [1, 3, 4, 4])
    y = make_tensor_value_info("y", [1, 48])
    shp = make_initializer("shp", [1, 48], elem_type=TensorProto.INT64)
    a = _run("Reshape", ["x", "shp"], ["y"], [x], [y], initializers=[shp])
    assert a.count_mac == 0 and a.count_alu == 0 and a.count_div == 0
    assert a.reshape_shape == [1, 48]


def test_div():
    x = make_tensor_value_info("x", [1, 3, 8, 8])
    z = make_tensor_value_info("z", [1, 3, 8, 8])
    y = make_tensor_value_info("y", [1, 3, 8, 8])
    a = _run("Div", ["x", "z"], ["y"], [x, z], [y])
    assert a.count_div == 1 * 3 * 8 * 8, a.count_div


def test_gather():
    data = make_tensor_value_info("data", [4, 10])
    y = make_tensor_value_info("y", [4, 2])
    idx = make_initializer("idx", [0, 3], elem_type=TensorProto.INT64)
    a = _run(
        "Gather", ["data", "idx"], ["y"], [data], [y],
        initializers=[idx], attributes={"axis": 1},
    )
    assert a.count_alu == 0 and a.count_div == 0
    assert a.gather_axis == 1


def test_slice():
    x = make_tensor_value_info("x", [20, 10, 5])
    y = make_tensor_value_info("y", [3, 10, 5])
    starts = make_initializer("s", [0], elem_type=TensorProto.INT64)
    ends = make_initializer("e", [3], elem_type=TensorProto.INT64)
    axes = make_initializer("ax", [0], elem_type=TensorProto.INT64)
    steps = make_initializer("st", [1], elem_type=TensorProto.INT64)
    a = _run(
        "Slice", ["x", "s", "e", "ax", "st"], ["y"], [x], [y],
        initializers=[starts, ends, axes, steps],
    )
    assert a.count_alu == 0
    assert a.slice_starts == [0] and a.slice_ends == [3]
    assert a.slice_axes == [0] and a.slice_steps == [1]


def test_pad_reflect():
    x = make_tensor_value_info("x", [1, 3, 224, 224])
    y = make_tensor_value_info("y", [1, 3, 232, 232])
    pads = make_initializer("p", [0, 0, 4, 4, 0, 0, 4, 4], elem_type=TensorProto.INT64)
    a = _run(
        "Pad", ["x", "p"], ["y"], [x], [y],
        initializers=[pads], attributes={"mode": "reflect"},
    )
    assert a.count_alu == 1 * 3 * 232 * 232, a.count_alu
    assert a.pad_mode == "reflect"
    assert a.pad_pads == [0, 0, 4, 4, 0, 0, 4, 4]


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _all_tests():
    return {
        name: obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    }


def main():
    tests = _all_tests()
    failures = []
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"FAIL  {name}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
