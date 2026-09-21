from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Conv")
class ConvNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Conv".

        * The op has MAC count of its output_dimension * kernel_shape
        * The op has ALU count of its output_dimension
        * The weight size of Conv is made of W and, optional B that is the second and third input of the node

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
        attributes.dilations = attr.get("dilations")
        attributes.group = attr.get("group")
        attributes.kernel_shape = attr.get("kernel_shape")
        attributes.pads = attr.get("pads")
        attributes.strides = attr.get("strides")

        # For Conv the second input is W
        attributes.sparsity = attributes.get_weight_sparsity(model, node.input[1])

        # For Conv the weight includes W and B, however B is only optional
        if len(node.input) == 3:
            attributes.weight_size = attributes.get_weight_size(
                model, node.input[1]
            ) + attributes.get_weight_size(model, node.input[2])
        else:
            attributes.weight_size = attributes.get_weight_size(model, node.input[1])

        # Calculating compute primitive. The number of input channels is the
        # -3 axis of the data tensor (N x C x H x W). When shape inference could
        # not resolve the data shape, fall back to deriving input channels from
        # the weight tensor W (shape: [out_ch, in_ch/group, kH, kW]).
        data_shape = attributes.input_dimension[0] if attributes.input_dimension else []
        group = attributes.group if attributes.group else 1
        if len(data_shape) >= 3:
            input_channels = data_shape[-3]
        else:
            weight_shape = attributes.get_weight_shape(model, node.input[1])
            if weight_shape is not None and len(weight_shape) >= 2:
                # W's in-channel axis is already divided by group
                input_channels = weight_shape[1] * group
            else:
                input_channels = group

        attributes.count_mac = (
            np.prod(attributes.output_dimension)
            * np.prod(attributes.kernel_shape)
            * (input_channels / group)
        )
        attributes.count_alu = np.prod(attributes.output_dimension)

        return attributes
