from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Mul")
def handler_mul_node(model, node):
    attributes = NodeAttributes(model, node)

    # Calculating compute primitive
    attributes.count_alu = np.prod(attributes.input_dimension)

    return attributes
