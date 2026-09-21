from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Slice")
class SliceNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Slice".

        Slice extracts a sub-tensor; it moves data but performs no arithmetic,
        so all compute primitive counts are zero.

        The slice parameters are represented differently across opset versions
        and are captured onto the structure either way:
        * opset <=9  : starts / ends / axes are list *attributes*
        * opset 10+  : starts / ends are inputs 1 / 2, and axes / steps are the
                       optional inputs 3 / 4. Values are captured when the
                       inputs are constant initializers.

        Any parameter provided as an initializer input is accounted toward
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

        # opset <=9 : starts / ends / axes are list attributes (no steps).
        starts = attr.get("starts")
        ends = attr.get("ends")
        axes = attr.get("axes")
        attributes.slice_starts = list(starts) if starts is not None else None
        attributes.slice_ends = list(ends) if ends is not None else None
        attributes.slice_axes = list(axes) if axes is not None else None

        # opset 10+ : starts, ends, (axes), (steps) are inputs 1..4. Capture
        # their values when they are constant initializers. Input order is
        # data(0), starts(1), ends(2), axes(3), steps(4).
        input_field_map = {
            1: "slice_starts",
            2: "slice_ends",
            3: "slice_axes",
            4: "slice_steps",
        }
        for index, field in input_field_map.items():
            if len(node.input) > index and node.input[index] != "":
                value = attributes.get_initializer_value(model, node.input[index])
                if value is not None:
                    setattr(attributes, field, value.tolist())

        # Data-movement only: no arithmetic primitives.

        # Slice parameter inputs may be provided as initializer coefficients.
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
