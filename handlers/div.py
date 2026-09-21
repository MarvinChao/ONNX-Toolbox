from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Div")
class DivNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Div". (element-wise C = A / B)

        * The op has DIV count of its output_dimension (one division per output
          element; the output is the broadcast result of A and B)
        * Legacy opset <=6 attributes "axis" and "broadcast" are captured onto
          the structure for consistency
        * Any operand provided as an initializer is accounted as a model
          coefficient in weight_size (mirrors Add / Sub / Mul)

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

        # Legacy opset <=6 broadcast attributes
        attributes.div_axis = attr.get("axis")
        attributes.div_broadcast = attr.get("broadcast")

        # Calculating compute primitive. Division counts follow the output
        # dimension (post-broadcast element count).
        attributes.count_div = np.prod(attributes.output_dimension)

        # Div operands could possibly contain coefficients
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
