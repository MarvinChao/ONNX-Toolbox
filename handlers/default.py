from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("default")
class DefaultHandler:
    def handle(self, model, node):
        """
        Default handler for unsupported op_types.
        Returns a minimal attribute dictionary with op_type and node name.

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        print(f"{node.name} has unsupported op_type [{node.op_type}]...")

        attributes = NodeAttributes(model, node, support=False)

        return attributes
