from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("MaxPool")
class MaxPoolNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "MaxPool".

        * The op has ALU count of its output_dimensionn (most processor should already have vector instructions to calculate max in single instruction)

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.output_dimension)

        return attributes
