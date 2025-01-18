from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Exp")
def handler_exp_node(model, node):
    attributes = NodeAttributes(model, node)

    # Calculating compute primitive
    attributes.count_exp = np.prod(attributes.input_dimension)

    return attributes
