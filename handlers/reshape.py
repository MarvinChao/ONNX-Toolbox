from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Reshape")
class ReshapeNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Reshape".

        Reshape only re-interprets the tensor layout; it moves data but performs
        no arithmetic, so all compute primitive counts are zero (consistent with
        Transpose / Concat).

        Attributes / inputs captured onto the structure:
        * allowzero (INT, opset 14+) : how a 0 in the target shape is treated
        * reshape_shape             : the target shape, taken from the legacy
                                      opset-1 "shape" attribute, or from the
                                      opset-5+ "shape" input when it is a
                                      constant initializer

        The "shape" input (int64 tensor), when it is an initializer, is counted
        toward weight_size as a model coefficient.

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

        # allowzero (opset 14+)
        attributes.allowzero = attr.get("allowzero")

        # Legacy opset-1 form: shape is a list attribute.
        legacy_shape = attr.get("shape")
        if legacy_shape is not None:
            attributes.reshape_shape = list(legacy_shape)

        # opset 5+ form: shape is the second input. Capture its value when it is
        # a constant initializer, and account it toward weight_size.
        if len(node.input) > 1 and node.input[1] != "":
            shape_value = attributes.get_initializer_value(model, node.input[1])
            if shape_value is not None:
                attributes.reshape_shape = shape_value.tolist()

        # Data-movement only: no arithmetic primitives.

        # The shape input may be provided as an initializer coefficient.
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
