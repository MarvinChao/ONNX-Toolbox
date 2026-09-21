"""
Regression test tooling for the ONNX-Toolbox.

This package provides hermetic (no network) utilities to build synthetic ONNX
graphs and run them through the toolbox analysis pipeline so that operator
handler behaviour can be locked down and regression-tested.

Modules:
    model_builder : helpers to construct small, deterministic ONNX graphs.
    regression    : run a model through the analysis pipeline and compare
                    per-node stats against a golden baseline.
"""

from utils.model_builder import (
    make_single_node_model,
    make_tensor_value_info,
    make_initializer,
)
from utils.regression import (
    RegressionRunner,
    analyze_model,
)

__all__ = [
    "make_single_node_model",
    "make_tensor_value_info",
    "make_initializer",
    "RegressionRunner",
    "analyze_model",
]
