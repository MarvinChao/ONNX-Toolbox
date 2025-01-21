from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Gemm")
class GemmNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Gemm".

        * The op has ALU count of its input_dimension
        * The op has DIV count of its output_dimension

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Determine the "effective" diemsnion of matrix A and B
        attr = {
            attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute
        }
        if attr.get("transA") != None:
            actual_A_dim = [
                attributes.input_dimension[0][1],
                attributes.input_dimension[0][0],
            ]
        else:
            actual_A_dim = [
                attributes.input_dimension[0][0],
                attributes.input_dimension[0][1],
            ]

        B_dim = attributes.get_weight_shape(model, node.input[1])
        if attr.get("transB") != None:
            actual_B_dim = [B_dim[1], B_dim[0]]
        else:
            actual_B_dim = [B_dim[0], B_dim[1]]

        # For Gemm the second input is B
        attributes.sparsity = attributes.get_weight_sparsity(model, node.input[1])

        # For Gemm the weight includes B and C
        attributes.weight_size = attributes.get_weight_size(
            model, node.input[1]
        ) + attributes.get_weight_size(model, node.input[2])

        # Calculating compute primitive
        attributes.count_mac = actual_A_dim[0] * actual_A_dim[1] * actual_B_dim[1]
        attributes.count_alu = np.prod(attributes.output_dimension)

        return attributes
