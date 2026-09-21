from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Gather")
class GatherNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Gather".

        Gather selects entries along "axis" of the data tensor using an index
        tensor and copies them into the output (output rank = q + (r - 1)).
        It is a pure indexed copy (e.g. embedding lookups), so it performs no
        arithmetic and all compute primitive counts are zero.

        Attributes captured onto the structure:
        * gather_axis (INT, default 0) : axis to gather on

        The "indices" input, when a constant initializer, is accounted toward
        weight_size as a model coefficient.

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

        # axis defaults to 0 when not specified
        axis = attr.get("axis")
        attributes.gather_axis = axis if axis is not None else 0

        # Data-movement only: no arithmetic primitives.

        # The indices input may be provided as an initializer coefficient.
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
