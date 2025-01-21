from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Mul")
class MulNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Mul".

        * The op has ALU count of its input_dimension (element-wise)

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.input_dimension)

        return attributes
