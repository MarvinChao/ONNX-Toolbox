from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Softplus")
class SoftplusNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Mish". ( Softplus(x) = ln(1+exp(x)) )

        * The op has ALU     count of its input_dimension
        * The op has EXP/LOG count of its input_dimension * 2

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.input_dimension)
        attributes.count_exp = np.prod(attributes.input_dimension) * 2

        return attributes
