from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Conv")
def handler_conv_node(model, node):
    attributes = NodeAttributes(model, node)

    # Parsing the op-specific attributes
    attr = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
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

    # Calculating compute primitive
    attributes.count_mac = np.prod(attributes.output_dimension) * np.prod(
        attributes.kernel_shape
    )
    attributes.count_alu = np.prod(attributes.output_dimension)

    return attributes
