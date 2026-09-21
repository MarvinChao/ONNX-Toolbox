import onnx
import numpy as np
from onnx import TensorProto
import pdb

# Create DataType mapping using definition in ONNX
onnx_dtype_map = {
    dtype_value: TensorProto.DataType.Name(dtype_value)
    for dtype_value in TensorProto.DataType.values()
}

# Manually define the number of bytes for each supported onnx datatype since this is not explicitly defined in https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L503~L551
onnx_dtype_byte_map = {
    "UNDEFINED": 0,
    "FLOAT": 4,
    "UINT8": 1,
    "INT8": 1,
    "UINT16": 2,
    "INT16": 2,
    "INT32": 4,
    "INT64": 8,
    "STRING": 0,
    "BOOL": 1,
    "FLOAT16": 2,
    "DOUBLE": 8,
    "UINT32": 4,
    "UINT64": 8,
    "COMPLEX64": 8,
    "COMPLEX128": 16,
    "BFLOAT16": 2,
    "FLOAT8E4M3FN": 1,
    "FLOAT8E4M3FNUZ": 1,
    "FLOAT8E5M2": 1,
    "FLOAT8E5M2FNUZ": 1,
    "UINT4": 0.5,
    "INT4": 0.5,
}


class NodeAttributes:
    """
    Extracting important node attributes for operation analysis

    Attributes:
    node_op_type (str):         The op_type of the node
    node_data_type (str):       The arithmetic data type of the node
    support (bool):             Flag indicating whether the op is supported (yet)
    node_name (str):            The name of the node
    input_dimension (list):     Input dimension
    output_dimension (list):    Output dimension
    input_size (int):           Input data bytes
    weight_size (int):          Weight data bytes
    output_size (int):          Output data bytes
    sparsity (float):           The sparsity of the weight (when applicable)
    count_mac (int):            Compute primitive count - Multiply-Accumulate
    count_alu (int):            Compute primitive count - ALU
    count_exp (int):            Compute primitive count - Exponent and Logarithm
    count_div (int):            Compute primitive count - Division
    count_trig (int):           Compute primitive count - Trigonometric functions
    count_sqrt (int):           Compute primitive count - Square root functions
    dilation (int):             Convolution attributes - dilation
    group (int):                Convolution attributes - group
    kernel_shape (int):         Convolution attributes - kernel_shape
    pads (int):                 Convolution attributes - pads
    strides (int):              Convolution attributes - strides
    auto_pad (str):             Pooling attributes - auto_pad
    ceil_mode (int):            Pooling attributes - ceil_mode
    count_include_pad (int):    Pooling attributes - count_include_pad
    clip_min (float):           Clip attributes - min (legacy opset <=6)
    clip_max (float):           Clip attributes - max (legacy opset <=6)
    reshape_shape (list):       Reshape attributes - target shape
    allowzero (int):            Reshape attributes - allowzero (opset 14+)
    div_axis (int):             Div attributes - axis (legacy opset <=6)
    div_broadcast (int):        Div attributes - broadcast (legacy opset <=6)
    gather_axis (int):          Gather attributes - axis
    slice_starts (list):        Slice attributes - starts
    slice_ends (list):          Slice attributes - ends
    slice_axes (list):          Slice attributes - axes
    slice_steps (list):         Slice attributes - steps
    pad_mode (str):             Pad attributes - mode
    pad_pads (list):            Pad attributes - pads
    pad_constant_value (float): Pad attributes - constant_value
    pad_axes (list):            Pad attributes - axes (opset 18+)
    """

    def __init__(self, model, node, support=True):
        if support == True:
            self.node_op_type = node.op_type
            self.node_data_type = self.get_tensor_type(model, node.input)
            self.support = True
            self.node_name = node.name
            self.input_dimension = self.get_input_shape(model, node)
            self.output_dimension = self.get_output_shape(model, node)
            self.input_size = self.get_input_size()
            self.weight_size = 0
            self.output_size = self.get_output_size()
            self.sparsity = 0
            # Tracking the primitive of operations
            self.count_mac = 0
            self.count_alu = 0
            self.count_exp = 0
            self.count_div = 0
            self.count_trig = 0
            self.count_sqrt = 0
            # Conv-specific attributes
            self.dilations = None
            self.group = None
            self.kernel_shape = None
            self.pads = None
            self.strides = None
            # Pooling-specific attributes
            self.auto_pad = None
            self.ceil_mode = None
            self.count_include_pad = None
            # Clip-specific attributes (legacy opset <=6 FLOAT attributes)
            self.clip_min = None
            self.clip_max = None
            # Reshape-specific attributes
            self.reshape_shape = None
            self.allowzero = None
            # Div-specific attributes (legacy opset <=6 attributes)
            self.div_axis = None
            self.div_broadcast = None
            # Gather-specific attributes
            self.gather_axis = None
            # Slice-specific attributes
            self.slice_starts = None
            self.slice_ends = None
            self.slice_axes = None
            self.slice_steps = None
            # Pad-specific attributes
            self.pad_mode = None
            self.pad_pads = None
            self.pad_constant_value = None
            self.pad_axes = None
            # Resize-specific attributes
            self.resize_mode = None
        else:
            self.node_op_type = node.op_type
            self.node_data_type = 0
            self.support = False
            self.node_name = node.name
            self.input_dimension = None
            self.output_dimension = None
            self.input_size = 0
            self.weight_size = 0
            self.output_size = 0
            self.sparsity = 0
            # Tracking the primitive of operations
            self.count_mac = 0
            self.count_alu = 0
            self.count_exp = 0
            self.count_div = 0
            self.count_trig = 0
            self.count_sqrt = 0
            # Conv-specific attributes
            self.dilations = None
            self.group = None
            self.kernel_shape = None
            self.pads = None
            self.strides = None
            # Pooling-specific attributes
            self.auto_pad = None
            self.ceil_mode = None
            self.count_include_pad = None
            # Clip-specific attributes (legacy opset <=6 FLOAT attributes)
            self.clip_min = None
            self.clip_max = None
            # Reshape-specific attributes
            self.reshape_shape = None
            self.allowzero = None
            # Div-specific attributes (legacy opset <=6 attributes)
            self.div_axis = None
            self.div_broadcast = None
            # Gather-specific attributes
            self.gather_axis = None
            # Slice-specific attributes
            self.slice_starts = None
            self.slice_ends = None
            self.slice_axes = None
            self.slice_steps = None
            # Pad-specific attributes
            self.pad_mode = None
            self.pad_pads = None
            self.pad_constant_value = None
            self.pad_axes = None
            # Resize-specific attributes
            self.resize_mode = None

    def is_tensor_name_initializer(self, model, tensor_name):
        initializers = model.graph.initializer

        for initializer in initializers:
            if initializer.name == tensor_name:
                return True
        return False

    def get_weight_sparsity(self, model, tensor_name):
        initializers = model.graph.initializer

        weight_tensor = None
        for initializer in initializers:
            if initializer.name == tensor_name:
                weight_tensor = initializer
                break

        if weight_tensor:
            weight_tensor = onnx.numpy_helper.to_array(weight_tensor)
            num_elements = np.prod(weight_tensor.shape)
            num_zeros = np.sum(weight_tensor == 0)
            sparsity = num_zeros / num_elements * 100
        else:
            sparsity = -1

        return sparsity

    def get_weight_size(self, model, tensor_name):
        initializers = model.graph.initializer

        weight_tensor = None
        for initializer in initializers:
            if initializer.name == tensor_name:
                weight_tensor = initializer
                break

        if weight_tensor:
            weight_tensor = onnx.numpy_helper.to_array(weight_tensor)
            num_elements = np.prod(weight_tensor.shape)
        else:
            num_elements = 0

        return num_elements * onnx_dtype_byte_map.get(
            onnx_dtype_map[self.node_data_type], 0
        )

    def get_weight_shape(self, model, tensor_name):
        initializers = model.graph.initializer

        weight_tensor = None
        for initializer in initializers:
            if initializer.name == tensor_name:
                weight_tensor = initializer
                break

        if weight_tensor:
            weight_tensor = onnx.numpy_helper.to_array(weight_tensor)
            return weight_tensor.shape
        else:
            return None

    def get_initializer_value(self, model, tensor_name):
        """
        Return the numpy value of an initializer by name, or None when the
        tensor is not a graph initializer (e.g. a dynamic/runtime input).
        """
        for initializer in model.graph.initializer:
            if initializer.name == tensor_name:
                return onnx.numpy_helper.to_array(initializer)
        return None

    def get_input_size(self):
        input_size = 0
        for input in self.input_dimension:
            input_size += np.prod(input)

        return input_size * onnx_dtype_byte_map.get(
            onnx_dtype_map[self.node_data_type], 0
        )

    def get_output_size(self):
        output_size = 0
        for output in self.output_dimension:
            output_size += np.prod(output)

        return output_size * onnx_dtype_byte_map.get(
            onnx_dtype_map[self.node_data_type], 0
        )

    def get_input_shape(self, model, node):
        """
        Locating the target tensor shape in either value_info, input, or output
        Each node could have multiple inputs so the function returns a list of all input dimensions
        """
        input_shape = []
        for input_name in node.input:
            tensor = self.find_tensor_by_name(model, input_name)
            if tensor is not None:
                shape = [
                    dim.dim_value
                    for dim in tensor.type.tensor_type.shape.dim
                    if dim.dim_value > 0
                ]
                input_shape.append(shape)

        return input_shape

    def get_output_shape(self, model, node):
        """
        Locating the target tensor shape in either value_info, input, or output
        Each node could have multiple outputs so the function returns a list of all output dimensions
        """
        output_shape = []
        for output_name in node.output:
            tensor = self.find_tensor_by_name(model, output_name)
            if tensor is not None:
                shape = [
                    dim.dim_value
                    for dim in tensor.type.tensor_type.shape.dim
                    if dim.dim_value > 0
                ]
                output_shape.append(shape)

        return output_shape

    def get_tensor_type(self, model, tensor_names):
        """
        Determine the tensor data type for a node using the non-initializer input tensor
        The DataType enum mapping can be found at https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L503~L551
        """
        for tensor_name in tensor_names:
            if not self.is_tensor_name_initializer(model, tensor_name):
                tensor = self.find_tensor_by_name(model, tensor_name)
                data_type = tensor.type.tensor_type.elem_type
                if data_type != None:
                    return data_type
                else:
                    return onnx.TensorProto.DataType.Value("UNDEFINED")

    def find_tensor_by_name(self, model, tensor_name):
        """
        Locating the target tensor from either value_info, input, output, or initializer
        """
        for tensor in model.graph.value_info:
            if tensor.name == tensor_name:
                return tensor
        for tensor in model.graph.input:
            if tensor.name == tensor_name:
                return tensor
        for tensor in model.graph.output:
            if tensor.name == tensor_name:
                return tensor

        #        initializers = model.graph.initializer

        #        weight_tensor = None
        #        for initializer in initializers:
        #            if initializer.name == tensor_name:
        #                weight_tensor = initializer
        #                return weight_tensor

        return None

    def to_dict(self):
        """
        This function is used as a formatted output NodeAttributes
        """
        return {
            "Operator Name": self.node_name,
            "Op Type": self.node_op_type,
            "Data Type": onnx_dtype_map[self.node_data_type],
            "Supported": self.support,
            "Input Dimensions": self.input_dimension,
            "Output Dimensions": self.output_dimension,
            "Dilation": self.dilations,
            "Group": self.group,
            "Kernel Shape": self.kernel_shape,
            "Pads": self.pads,
            "Strides": self.strides,
            "Auto Pad": self.auto_pad,
            "Ceil Mode": self.ceil_mode,
            "Count Include Pad": self.count_include_pad,
            "Clip Min": self.clip_min,
            "Clip Max": self.clip_max,
            "Reshape Shape": self.reshape_shape,
            "Allow Zero": self.allowzero,
            "Div Axis": self.div_axis,
            "Div Broadcast": self.div_broadcast,
            "Gather Axis": self.gather_axis,
            "Slice Starts": self.slice_starts,
            "Slice Ends": self.slice_ends,
            "Slice Axes": self.slice_axes,
            "Slice Steps": self.slice_steps,
            "Pad Mode": self.pad_mode,
            "Pad Pads": self.pad_pads,
            "Pad Constant Value": self.pad_constant_value,
            "Pad Axes": self.pad_axes,
            "Resize Mode": self.resize_mode,
            "MAC Count": self.count_mac,
            "ALU Count": self.count_alu,
            "EXP Count": self.count_exp,
            "DIV Count": self.count_div,
            "TRIG Count": self.count_trig,
            "SQRT Count": self.count_sqrt,
            "Input Size (bytes)": self.input_size,
            "Weight Size (bytes)": self.weight_size,
            "Output Size (bytes)": self.output_size,
            "Sparsity": self.sparsity,
        }
