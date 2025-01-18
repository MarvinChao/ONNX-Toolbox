from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np

@register_node_handler("MaxPool")
def handler_maxpool_node(model, node):
    attributes = NodeAttributes(model, node)

    # Calculating compute primitive
    # Note: most processor should already have vector instructions to calculate max in one single instruction
    attributes.count_alu = np.prod(attributes.output_dimension)

    return attributes

