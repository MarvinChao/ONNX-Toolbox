from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("InstanceNormalization")
class InstanceNormNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "InstanceNormalization"

        * Layer normalization is per-batch, per-channel

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_mac = np.prod(attributes.input_dimension)
        attributes.count_alu = np.prod(attributes.input_dimension) * 6
        attributes.count_div = attributes.input_dimension[
            0
        ] * attributes.input_dimension[1] * 2 + np.prod(attributes.input_dimension)
        attributes.count_sqrt = (
            attributes.input_dimension[0] * attributes.input_dimension[1]
        )

        # Add inputs could possibly contains coefficients
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
