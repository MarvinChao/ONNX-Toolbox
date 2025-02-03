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
from MemTracker import MemTracker

import pdb


def is_chainable(op_type):
    """
    Example logic for which ops can be chained together without explicit DRAM flush
    """
    chainable_ops = {
        "Relu",
        "Sigmoid",
        "Tanh",
        "Add",
        "Mul",
        "Mish",
        "Transpose",
        "LeakyRelu",
        "Concat",
    }
    return op_type in chainable_ops


class ModelStats:
    """
    Collect native model statistics here

    Attributes:
    onnx_filename (string):     The input arguments
    model (Class):              Loaded ONNX model
    ops_attributes (list):      The list of attributes of all ops in the ONNX model
    ref_count (dict):           The tensor reference count used to track local memory usage
    tensor_size (dict):         The size of all tensors in the ONNX model
    local_memory_size (int):    The size of local SRAM
    verbose (bool):             Verbose output flag
    """

    def __init__(self, args):
        self.verbose = args.verbose
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

        self.local_memory_size = args.memory
        self.add_memory_tracker()

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

    def partition_model(self):
        """
        Returns a list of segments, where each segment is a list of nodes
        that can be chained together before forcing a store to DRAM.
        """
        segments = []
        current_segment = []

        for node in self.model.graph.node:
            # Start a new segment when the node is unchainable
            if not is_chainable(node.op_type):
                segments.append(current_segment)
                current_segment = []
                current_segment.append(node)
            else:
                current_segment.append(node)

            # Adding leftover segment to the segment list
            if current_segment:
                segments.append(current_segment)

            return segments

    def add_memory_tracker(self):
        mem_tracker = MemTracker(
            self.model, self.tensor_size, self.ref_count, self.local_memory_size
        )

        num_nodes = len(self.model.graph.node)
        for i, node in enumerate(self.model.graph.node):
            if i == num_nodes - 1:
                next_node_chainable = False
            else:
                # The next node needs to be a chainable op and it has to be connected to current node
                next_node = self.model.graph.node[i + 1]
                next_node_chainable = (
                    is_chainable(next_node.op_type)
                    and set(node.output).intersection(set(next_node.input)) != {}
                )

            node_stats = mem_tracker.process_node(node, next_node_chainable)
            self.ops_attributes[i]["bytes_loaded"] = node_stats["bytes_loaded"]
            self.ops_attributes[i]["bytes_stored"] = node_stats["bytes_stored"]
            if self.verbose:
                self.ops_attributes[i]["Local SRAM footprint"] = node_stats["footprint"]
                self.ops_attributes[i]["Next Node Chainable"] = node_stats[
                    "next_node_chainable"
                ]

        mem_tracker.finalize()

    def save_model(self):
        print(
            f"Export model to {os.path.splitext(os.path.basename((self.onnx_filename))[0] + '_opt.onnx')}"
        )
        onnx.save(
            self.model,
            os.path.splitext(os.path.basename(self.onnx_filename))[0] + "_opt.onnx",
        )

    def generate_report(self):
        report_generator = ReportGenerator(self.ops_attributes, self.xlsx_filename)
        print(f"Generate model analysis report to {self.xlsx_filename}")
        report_generator.write_xlsx()
