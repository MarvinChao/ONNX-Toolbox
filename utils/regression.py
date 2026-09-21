"""
Run ONNX models through the toolbox analysis pipeline and compare the per-node
statistics against a golden baseline.

This mirrors the dispatch path used by onnx_analysis.ModelStats.parse_model
(shape inference -> per-node handler dispatch -> to_dict) without depending on
the full ModelStats object, so it can be exercised on synthetic graphs in a
hermetic test.
"""

import json
import os

import onnx

# Import handlers to populate the registry as a side effect.
import handlers  # noqa: F401
from node_registry import get_handler


def _try_shape_inference(model):
    """
    Best-effort shape inference, mirroring onnx_analysis.shape_infer_model:
    prefer ORT SymbolicShapeInference, fall back to onnx.shape_inference.
    """
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        return SymbolicShapeInference.infer_shapes(model)
    except Exception:
        try:
            return onnx.shape_inference.infer_shapes(model)
        except Exception:
            return model


def analyze_model(model, run_shape_inference=False, check=False):
    """
    Dispatch every node in the model through its handler and collect the
    per-node attribute dicts (to_dict output).

    Args:
        model (onnx.ModelProto):    The model to analyze.
        run_shape_inference (bool): Whether to run shape inference first.
        check (bool):               Whether to run onnx.checker first.

    Returns:
        list[dict]: One to_dict() per node, in graph order.
    """
    if check:
        onnx.checker.check_model(model)
    if run_shape_inference:
        model = _try_shape_inference(model)

    stats = []
    for node in model.graph.node:
        handler = get_handler(node.op_type)
        stats.append(handler.handle(model, node).to_dict())
    return stats


def _normalize(value):
    """
    Normalize a stat value for stable JSON comparison. numpy scalars/arrays and
    tuples are converted to plain python types.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    return value


def _normalize_stats(stats):
    return [{k: _normalize(v) for k, v in row.items()} for row in stats]


class RegressionRunner:
    """
    Save and compare golden baselines of per-node analysis stats.

    Usage:
        runner = RegressionRunner(golden_dir)
        stats = analyze_model(model)
        runner.save("mymodel", stats)        # regenerate baseline
        diffs = runner.compare("mymodel", stats)  # returns list of differences
    """

    def __init__(self, golden_dir):
        self.golden_dir = golden_dir

    def _path(self, name):
        return os.path.join(self.golden_dir, f"{name}.json")

    def save(self, name, stats):
        """Write the golden baseline for a named case."""
        os.makedirs(self.golden_dir, exist_ok=True)
        with open(self._path(name), "w") as f:
            json.dump(_normalize_stats(stats), f, indent=2, sort_keys=True)

    def load(self, name):
        """Load a golden baseline, or None if it does not exist."""
        path = self._path(name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def compare(self, name, stats):
        """
        Compare current stats against the golden baseline.

        Returns:
            list[str]: Human-readable differences. Empty means a match.
                       A missing baseline is reported as a single difference.
        """
        golden = self.load(name)
        current = _normalize_stats(stats)

        if golden is None:
            return [f"No golden baseline found for '{name}'. Run with save to create it."]

        diffs = []
        if len(golden) != len(current):
            diffs.append(
                f"node count changed: golden={len(golden)} current={len(current)}"
            )

        for i, (g_row, c_row) in enumerate(zip(golden, current)):
            keys = set(g_row) | set(c_row)
            for key in sorted(keys):
                g_val = g_row.get(key, "<missing>")
                c_val = c_row.get(key, "<missing>")
                if g_val != c_val:
                    op = c_row.get("Op Type", g_row.get("Op Type", "?"))
                    diffs.append(
                        f"node[{i}] ({op}) '{key}': golden={g_val!r} current={c_val!r}"
                    )
        return diffs
