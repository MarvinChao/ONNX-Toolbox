from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("LeakyRelu")
def handler_LeakyRelu_node(model, node):
    attributes = NodeAttributes(model, node)

    # Calculating compute primitive
    # Theoretically this could take 2~3 (cmp, mul, and assign) instructions
    attributes.count_alu = np.prod(attributes.input_dimension) * 2.5

    return attributes
