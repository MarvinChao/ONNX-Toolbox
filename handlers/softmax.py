from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Softmax")
def handler_softmax_node(model, node):
    attributes = NodeAttributes(model, node)

    attr = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    axis = attr.get("axis")

    # Calculating compute primitive
    attributes.count_alu = attributes.input_dimension[0][axis]
    attributes.count_exp = np.prod(attributes.input_dimension)
    attributes.count_div = attributes.input_dimension[0][axis]

    return attributes
