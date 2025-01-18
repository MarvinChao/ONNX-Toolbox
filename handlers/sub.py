from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Sub")
def handler_sub_node(model, node):
    attributes = NodeAttributes(model, node)

    # Calculating compute primitive
    attributes.count_alu = np.prod(attributes.output_dimension)

    # Add inputs could possibly contains coefficients
    for tensor_name in node.input:
        if attributes.is_tensor_name_initializer(model, tensor_name):
            attributes.weight_size += attributes.get_weight_size(model, tensor_name)

    return attributes
