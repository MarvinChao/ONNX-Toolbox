import os
import onnx
from onnx import TensorProto
import numpy as np
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from node_registry import ONNX_OPS_REGISTRY, register_node_handler, get_handler
from node_attributes import NodeAttributes
import handlers
from gen_report import ReportGenerator

import pdb


class ModelStats:
    """
    Collect native model statistics here

    Attributes:
    onnx_filename (str):        The filename of onnx model
    ONNX_OPS_REGISTRY (list):   Registry of all the ops handler
    model (Class):              Loaded ONNX model
    ops_attributes (list):      The list of attributes of all ops in the ONNX model
    """

    def __init__(self, onnx_filename):
        self.onnx_filename = onnx_filename
        #        self.ONNX_OPS_REGISTRY = ONNX_OPS_REGISTRY
        self.model = None
        self.xlsx_filename = (
            os.path.splitext(os.path.basename(onnx_filename))[0] + ".xlsx"
        )
        self.ops_attributes = []
        self.unsupported_ops = {}
        self.gen_report = None

        self.load_model()
        self.shape_infer_model()
        self.parse_model()

    def load_model(self):
        print(f"Loading ONNX model: {self.onnx_filename}")
        self.model = onnx.load(self.onnx_filename)
        onnx.checker.check_model(self.model)

    def shape_infer_model(self):
        print(f"Shape-infering the model...")
        if self.model:
            try:
                self.model = SymbolicShapeInference.infer_shapes(self.model)
            except Exception as e:
                print(
                    f"ORT SymbolicShapeInference failed with error: {e}, falling back to onnx.shape_inference.infer_shapes() method"
                )
                self.model = onnx.shape_inference.infer_shapes(self.model)

    def parse_model(self):
        for node in self.model.graph.node:
            ops_handler = get_handler(node.op_type)
            #            ops_handler = self.ONNX_OPS_REGISTRY.get(
            #                node.op_type, self.ONNX_OPS_REGISTRY["default"]
            #            )
            self.ops_attributes.append(ops_handler.handle(self.model, node).to_dict())

    def save_model(self):
        onnx.save(
            self.model,
            os.path.splitext(os.path.basename(self.onnx_filename))[0] + "_opt.onnx",
        )

    def generate_report(self):
        report_generator = ReportGenerator(self.ops_attributes, self.xlsx_filename)
        report_generator.write_xlsx()
