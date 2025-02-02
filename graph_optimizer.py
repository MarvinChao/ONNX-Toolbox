import os
import copy
import argparse
import onnx
import onnxoptimizer
import onnxruntime as ort

from onnxsim import simplify
import numpy as np

from toolbox_optimizer import ToolboxOptimizer


class GraphOptimizer:
    def __init__(self, onnx_filename, method, level, export):
        self.onnx_filename = onnx_filename
        self.method = method
        self.level = level
        self.export = export
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading ONNX model: {self.onnx_filename}")
        self.model = onnx.load(self.onnx_filename)
        onnx.checker.check_model(self.model)

    def export_model(self):
        export_path = (
            os.path.splitext(os.path.basename(self.onnx_filename))[0] + "_opt.onnx",
        )
        print(f"Export optimized ONNX model: {export_path[0]}")
        onnx.save(
            self.model,
            export_path[0],
        )

    def check_model(self):
        """
        Check the integrity of the ONNX model
        """
        try:
            onnx.checker.check_model(self.model)
            print("ONNX Model Check: Passed")
        except onnx.checker.ValidationError as e:
            print(f"ONNX Model Check: Failed: {e}")
            return False

        if not self.model.graph.input or not self.model.graph.output:
            print("Model Input/Output Check: Failed")
            return False
        print("Model Input/Output Check: Passed")

        # ORT inference test
        temp_model_path = "temp_model.onnx"
        onnx.save(self.model, temp_model_path)
        session = ort.InferenceSession(temp_model_path)

        # Generate random input matching the first input shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_data = np.random.randn(
            *[dim if isinstance(dim, int) else 1 for dim in input_shape]
        ).astype(np.float32)

        try:
            session.run(None, {input_name: input_data})
            print("Model Inference Check: Passed")
        except Exception as e:
            print(f"Model Inference Check: Failed: {e}")
            return False

        # Check ONNX version
        print(f"ONNX IR Version: {self.model.ir_version}")
        print(f"Produced By: {self.model.producer_name}")
        print(f"Version: {self.model.producer_version}")

        return True

    def simplify_model(self):
        """
        Removing redundant operators in the model
        """
        self.model, check = simplify(self.model)

        if check:
            print("ONNX Simplifier successfully simplified the model.")
        else:
            print("ONNX Simplifier failed.")
            return False

        return True

    def onnx_optimizer(self):
        """
        Apply ONNX built-in graph optimizations
        """
        passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_transpose",
            "eliminate_nop_pad",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
        ]
        self.model = onnxoptimizer.optimize(self.model, passes)
        print("ONNX built-in optimizations applied.")

    def ort_optimizer(self):
        """
        Enable ONNX Runtime optimizations
        ORT and ONNX do no0t share the same model format so we have to use an intermediate file to carry over
        """
        temp_model_path = "temp_model.onnx"

        try:
            onnx.save(self.model, temp_model_path)

            sess_options = ort.SessionOptions()
            if self.level == "all":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
            elif self.level == "basic":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                )
            elif self.level == "extended":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                )

            sess_options.optimized_model_filepath = temp_model_path
            ort_session = ort.InferenceSession(temp_model_path, sess_options)
            print(f"ONNX Runtime {self.level.upper()} optimizations applied.")

            self.model = onnx.load(temp_model_path)
        finally:
            os.remove(temp_model_path)

    def toolbox_optimizer(self):
        """
        Apply custom implementation of model optimization not found in public tools
        """
        optimizer = ToolboxOptimizer(self.model)
        optimizer.apply()
        self.model = copy.deepcopy(optimizer.model)
        print(f"ONNX-Toolbox optimizations applied")

    def execute(self):
        """
        Execute model optimization steps
        """
        print(f"===== Checking input model integrity =====")
        self.check_model()

        print(f"===== Perform assigned optimizations {self.method} =====")
        match self.method:
            case "onnxsim":
                self.simplify_model()
            case "onnx":
                self.onnx_optimizer()
            case "ort":
                self.ort_optimizer()
            case "onnx-toolbox":
                self.toolbox_optimizer()
            case "all":
                self.simplify_model()
                self.onnx_optimizer()
                self.ort_optimizer()
                self.toolbox_optimizer()
            case _:
                return

        print(f"===== Export optimized model: {self.export} =====")
        self.export_model()


def check_args(args):
    # Check input file
    if os.path.splitext(args.input)[1].lower() != ".onnx":
        print(f"Input file {args.input} is not an ONNX file")
        return False

    # Check optimizations method
    supported_methods = ["onnxsim", "onnx", "ort", "onnx-toolbox", "all"]
    if args.method not in supported_methods:
        print(f"Method {args.method} is not supported")
        return False

    # Check optimizations level
    supported_levels = ["basic", "extend", "ort", "all"]
    if args.level not in supported_levels:
        print(f"Level {args.level} is not supported")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="ONNX Graph Optimization Tool")

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input ONNX model filename"
    )

    parser.add_argument(
        "--method",
        "-m",
        type=str,
        required=True,
        default="all",
        help="Choose optimizer to run [onnxsim|onnx|ort|onnx-toolbox|all]",
    )

    parser.add_argument(
        "--level",
        "-l",
        type=str,
        required=False,
        default="all",
        help="Specify optimization level [basic|extend|all]",
    )

    parser.add_argument(
        "--export",
        "-e",
        action="store_true",
        required=False,
        help="Export optimized ONNX model",
    )

    args = parser.parse_args()

    if check_args(args) != True:
        print("Input arguments check failed. Please double check!!!")
        return

    graph_optimizer = GraphOptimizer(args.input, args.method, args.level, args.export)

    graph_optimizer.execute()


if __name__ == "__main__":
    main()
