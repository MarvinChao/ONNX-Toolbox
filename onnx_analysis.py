import os
import onnx
from onnx import TensorProto, mapping
import numpy as np
import argparse
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
    onnx_filename (string):     The input arguments
    model (Class):              Loaded ONNX model
    ops_attributes (list):      The list of attributes of all ops in the ONNX model
    ref_count (dict):           The tensor reference count used to track local memory usage
    tensor_size (dict):         The size of all tensors in the ONNX model
    """

    def __init__(self, args):
        self.onnx_filename = args.input
        self.model = self.load_model()
        self.xlsx_filename = (
            os.path.splitext(os.path.basename(self.onnx_filename))[0] + ".xlsx"
        )
        self.ops_attributes = []
        self.unsupported_ops = {}

        self.check_model()
        self.shape_infer_model()
        self.parse_model()

        self.ref_count = self.build_ref_count_map()
        self.tensor_size = self.build_tensor_size_map()

        self.track_local_foorprint()

    def load_model(self):
        print(f"Loading ONNX model: {self.onnx_filename}")
        return onnx.load(self.onnx_filename)

    def check_model(self):
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
            self.ops_attributes.append(ops_handler.handle(self.model, node).to_dict())

    def build_ref_count_map(self):
        ref_count = {}

        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name:
                    ref_count[input_name] = ref_count.get(input_name, 0) + 1
        return ref_count

    def get_tensor_type(self, value_info):
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        return elem_type

    def get_tensor_shape(self, value_info):
        tensor_type = value_info.type.tensor_type
        shape = []
        for d in tensor_type.shape.dim:
            if d.dim_value is not None and d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(1)
        return shape

    def get_tensor_size(self, elem_type, shape):
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type]
        item_size = np.dtype(np_dtype).itemsize
        num_elements = 1
        for s in shape:
            num_elements *= s
        return item_size * num_elements

    def build_tensor_size_map(self):
        size_map = {}

        for value_info in self.model.graph.value_info:
            elem_type = self.get_tensor_type(value_info)
            shape = self.get_tensor_shape(value_info)
            name = value_info.name
            size_map[name] = self.get_tensor_size(elem_type, shape)

        for input_info in self.model.graph.input:
            elem_type = self.get_tensor_type(input_info)
            shape = self.get_tensor_shape(input_info)
            name = input_info.name
            size_map[name] = self.get_tensor_size(elem_type, shape)

        for output_info in self.model.graph.output:
            elem_type = self.get_tensor_type(output_info)
            shape = self.get_tensor_shape(output_info)
            name = output_info.name
            size_map[name] = self.get_tensor_size(elem_type, shape)

        for init in self.model.graph.initializer:
            if init.name not in size_map:
                arr = onnx.numpy_helper.to_array(init)
                size_map[init.name] = arr.nbytes

        return size_map

    def track_local_foorprint(self):
        current_footprint = 0
        max_footprint = 0
        allocated_tensors = set()

        for i, node in enumerate(self.model.graph.node):
            # 1. Allocate input memory
            for input_name in node.input:
                if not input_name:
                    continue

            if input_name not in allocated_tensors:
                if input_name in self.tensor_size:
                    current_footprint += self.tensor_size[input_name]
                    allocated_tensors.add(input_name)

            # 2. Allocate output memory
            for output_name in node.output:
                if not output_name:
                    continue

                if output_name in self.tensor_size:
                    current_footprint += self.tensor_size[output_name]
                    allocated_tensors.add(output_name)

                    # Update peak usage
                    if current_footprint > max_footprint:
                        max_footprint = current_footprint

            self.ops_attributes[i]["SRAM Footprint"] = current_footprint

            # 3. After the node finishes, free input/weight if ref_count hits 0
            for input_name in node.input:
                if not input_name:
                    continue

                if input_name in self.ref_count:
                    self.ref_count[input_name] -= 1
                    if self.ref_count[input_name] == 0:
                        if input_name in allocated_tensors:
                            current_footprint -= self.tensor_size[input_name]
                            allocated_tensors.remove(input_name)

            return max_footprint

    def save_model(self):
        onnx.save(
            self.model,
            os.path.splitext(os.path.basename(self.onnx_filename))[0] + "_opt.onnx",
        )

    def generate_report(self):
        report_generator = ReportGenerator(self.ops_attributes, self.xlsx_filename)
        report_generator.write_xlsx()
