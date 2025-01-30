from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np
import onnx


@register_node_handler("Resize")
class ResizeNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Resize".

        * The "mode" and "scales" will determine the compute cost of this node
        * Here we are assume a more generic, less optimized compute cost
        * mode = "nearest": ALU count is output_dimension
        * mode = "linear":  ALU count is output_dimension * 23
        * mode = "cubic":  ALU count is output_dimension * 109

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
        attributes.resize_mode = attr.get("mode").decode()

        # Calculating compute primitive
        if attributes.resize_mode == "nearest":
            attributes.count_alu = np.prod(attributes.output_dimension)
        elif attributes.resize_mode == "linear":
            attributes.count_alu = np.prod(attributes.output_dimension) * 23
        elif attributes.resize_mode == "cubic":
            attributes.count_alu = np.prod(attributes.output_dimension) * 109

        # Add inputs could possibly contains coefficients
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
