from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("AveragePool")
class AveragePoolNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "AveragePool".

        AveragePool slides a window over the input and, for each output element,
        averages the values inside that window (average = sum(window) / count).

        * The op has ALU count of output_dimension * (kernel_size - 1), where
          kernel_size = prod(kernel_shape). Averaging a window of N elements
          takes N-1 additions; dilations do not change how many elements are
          summed, only which input elements are picked.
        * The op has DIV count of output_dimension (one division per output
          element to turn the accumulated sum into an average).

        Supported ONNX attributes (see onnx__AveragePool):
            auto_pad, ceil_mode, count_include_pad, dilations, kernel_shape,
            pads, strides

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Parsing the op-specific attributes
        attr = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }

        # auto_pad is stored as bytes in the ONNX protobuf, decode when present
        auto_pad = attr.get("auto_pad")
        attributes.auto_pad = auto_pad.decode() if auto_pad is not None else None
        attributes.ceil_mode = attr.get("ceil_mode")
        attributes.count_include_pad = attr.get("count_include_pad")
        attributes.dilations = attr.get("dilations")
        attributes.kernel_shape = attr.get("kernel_shape")
        attributes.pads = attr.get("pads")
        attributes.strides = attr.get("strides")

        # Number of elements accumulated per output element. kernel_shape is a
        # required attribute; guard defensively in case it is absent.
        kernel_size = np.prod(attributes.kernel_shape) if attributes.kernel_shape else 1

        # Calculating compute primitive
        num_outputs = np.prod(attributes.output_dimension)
        attributes.count_alu = num_outputs * (kernel_size - 1)
        attributes.count_div = num_outputs

        return attributes
