from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Mish")
class MishNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Mish". ( Mish(x) = x * tanh( ln(1+exp(x))) )

        * The op has ALU     count of its input_dimension * 2 (add and mul)
        * The op has EXP/LOG count of its input_dimension * 2 (exp and log)
        * The op has DIV     count of its input_dimension
        * The op has TRIG    count of its input_dimension (tanh)

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
        attributes.count_div = np.prod(attributes.input_dimension)
        attributes.count_trig = np.prod(attributes.input_dimension)

        return attributes
