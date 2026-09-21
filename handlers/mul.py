from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("Mul")
class MulNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Mul". (element-wise)

        * The op has ALU count of its output_dimension (post-broadcast element
          count; using output_dimension avoids ragged multi-input shapes)
        * Since Mul could contain initializer, they will be considered as model
          coefficient

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # Calculating compute primitive
        attributes.count_alu = np.prod(attributes.output_dimension)

        # Mul inputs could possibly contain coefficients
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
