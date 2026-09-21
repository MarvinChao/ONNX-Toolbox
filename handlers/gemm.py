from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Gemm")
class GemmNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Gemm". ( Y = alpha * A' * B' + beta * C )

        * The op has MAC count of M * K * N (from the effective A and B dims,
          honoring transA / transB)
        * The op has ALU count of its output_dimension

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

        # A is expected to be a 2-D matrix [M, K]. Some exporters feed a 1-D
        # tensor (a single row, M implicitly 1) or leave the shape unresolved;
        # normalize to a 2-D shape so indexing is always safe.
        A_shape = attributes.input_dimension[0] if attributes.input_dimension else []
        if len(A_shape) >= 2:
            A_dim = [A_shape[0], A_shape[1]]
        elif len(A_shape) == 1:
            A_dim = [1, A_shape[0]]
        else:
            A_dim = [1, 1]

        if attr.get("transA"):
            actual_A_dim = [A_dim[1], A_dim[0]]
        else:
            actual_A_dim = [A_dim[0], A_dim[1]]

        # B is the second input (weight matrix). Normalize to 2-D and honor the
        # transB flag only when it is set to a truthy value.
        B_shape = attributes.get_weight_shape(model, node.input[1])
        if B_shape is not None and len(B_shape) >= 2:
            B_dim = [B_shape[0], B_shape[1]]
        elif B_shape is not None and len(B_shape) == 1:
            B_dim = [B_shape[0], 1]
        else:
            B_dim = [1, 1]

        if attr.get("transB"):
            actual_B_dim = [B_dim[1], B_dim[0]]
        else:
            actual_B_dim = [B_dim[0], B_dim[1]]

        # For Gemm the second input is B
        attributes.sparsity = attributes.get_weight_sparsity(model, node.input[1])

        # For Gemm the weight includes B and the optional bias C (third input)
        attributes.weight_size = attributes.get_weight_size(model, node.input[1])
        if len(node.input) > 2 and node.input[2] != "":
            attributes.weight_size += attributes.get_weight_size(model, node.input[2])

        # Calculating compute primitive
        attributes.count_mac = actual_A_dim[0] * actual_A_dim[1] * actual_B_dim[1]
        attributes.count_alu = np.prod(attributes.output_dimension)

        return attributes
