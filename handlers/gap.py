from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("GlobalAveragePool")
class GAPNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "GlobalAveragePool".

        * The op has ALU count of its input_dimension
        * The op has DIV count of its output_dimension

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.input_dimension)
        attributes.count_div = np.prod(attributes.output_dimension)

        return attributes
