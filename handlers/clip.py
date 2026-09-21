from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Clip")
class ClipNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Clip". ( Clip = min( max(x, min_val), max_val ) )

        This is a very common bounded activation (e.g. ReLU6 is Clip(x, 0, 6)).

        The min/max bounds are represented differently across opset versions:
        * opset 11+ : "min" and "max" are optional scalar *inputs*
                      (node.input[1] and node.input[2]); an omitted optional
                      input is either absent or an empty string "".
        * opset <=6 : "min" and "max" are optional FLOAT *attributes*.

        Either way the bound values are captured onto clip_min / clip_max so
        the structure is consistent. For the input form, a value is only
        recoverable when the bound is a constant initializer; a dynamic
        (runtime) bound is still counted but its value stays None.

        Compute cost:
        * Each bound that is present costs one compare/select per element.
          So the op has ALU count of input_dimension * (number of bounds
          present) -- 2 for a fully bounded Clip (e.g. ReLU6), 1 when only
          one bound is given, 0 for a degenerate pass-through Clip.
        * Any bound provided as an initializer input is accounted as a model
          coefficient in weight_size.

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

        # Start from the legacy opset <=6 FLOAT attributes (None when absent).
        attributes.clip_min = attr.get("min")
        attributes.clip_max = attr.get("max")

        # opset 11+ : min/max are optional inputs at index 1 (min) and 2 (max).
        # An omitted optional input is either missing entirely or an empty
        # string. When the bound is a constant initializer, capture its value
        # onto the same clip_min / clip_max fields for consistency.
        min_present = len(node.input) > 1 and node.input[1] != ""
        max_present = len(node.input) > 2 and node.input[2] != ""
        if min_present:
            value = attributes.get_initializer_value(model, node.input[1])
            if value is not None:
                attributes.clip_min = value.item() if value.size == 1 else value
        if max_present:
            value = attributes.get_initializer_value(model, node.input[2])
            if value is not None:
                attributes.clip_max = value.item() if value.size == 1 else value

        # Count how many bounds are actually applied. A bound contributes one
        # compare/select per element.
        num_bounds = 0
        # opset 11+ input-form bounds
        if min_present:
            num_bounds += 1
        if max_present:
            num_bounds += 1
        # opset <=6 attribute-form bounds (only when no input form was used)
        if not min_present and attr.get("min") is not None:
            num_bounds += 1
        if not max_present and attr.get("max") is not None:
            num_bounds += 1

        # Calculating compute primitive. The first input is the data tensor;
        # the optional min/max inputs are scalars and are not part of the
        # element-wise workload.
        attributes.count_alu = np.prod(attributes.input_dimension[0]) * num_bounds

        # The optional min/max bounds could be provided as initializers
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
