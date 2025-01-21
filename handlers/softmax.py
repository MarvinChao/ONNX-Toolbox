from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Softmax")
class SoftmaxNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Softmax". ( Softmax = exp(x_n) / sum( exp(x_n) ) )

        * The op has ALU count of its input_dimension on pre-defined axis
        * The op has EXP count of its input_dimension
        * The op has DIV count of its input_dimension on pre-defined axis

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        attr = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }
        axis = attr.get("axis")

        # Calculating compute primitive
        attributes.count_alu = attributes.input_dimension[0][axis]
        attributes.count_exp = np.prod(attributes.input_dimension)
        attributes.count_div = attributes.input_dimension[0][axis]

        return attributes
