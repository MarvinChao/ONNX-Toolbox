from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("LeakyRelu")
class LeakyReluNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Gemm".

        * The op has ALU count of its input_dimension * 2.5 (cmp/mul/assign instructions depend on its sign )

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.input_dimension) * 2.5

        return attributes
