from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Relu")
class ReluNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Relu".

        * The op has ALU count of its input_dimension * 0.5 (only for positive input value)

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.input_dimension) * 0.5

        return attributes
