from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Sigmoid")
class SigmoidNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Sigmoid". ( Sigmoid = 1 / ( 1 + exp(-x) ) )

        * The op has ALU count of its input_dimension
        * The op has EXP count of its input_dimension
        * The op has DIV count of its input_dimension

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.input_dimension)
        attributes.count_exp = np.prod(attributes.input_dimension)
        attributes.count_div = np.prod(attributes.input_dimension)

        return attributes
