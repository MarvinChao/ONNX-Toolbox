from node_registry import register_node_handler
from node_attributes import NodeAttributes
import onnx
import numpy as np


@register_node_handler("Pad")
class PadNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "Pad".

        Pad writes a border of constant / reflected / edge / wrapped values
        around the data tensor. Modeled as one write (ALU op) per output
        element, i.e. ALU count equals output_dimension. This treats the border
        fill as the dominant cost and is a conservative, HW-agnostic estimate.

        Attributes / inputs captured onto the structure:
        * pad_mode (STRING, default "constant")
        * pad_pads : from the opset <=2 "pads" attribute, or the opset 11+
                     "pads" input when it is a constant initializer
        * pad_constant_value : from the opset <=2 "value" attribute, or the
                               opset 11+ "constant_value" input when constant
        * pad_axes : the optional opset 18+ "axes" input when constant

        Any pad parameter provided as an initializer input is accounted toward
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

        # mode is stored as bytes in the ONNX protobuf; decode when present.
        mode = attr.get("mode")
        if mode is not None:
            attributes.pad_mode = mode.decode() if isinstance(mode, bytes) else mode
        else:
            attributes.pad_mode = "constant"

        # opset <=2 : pads / value are attributes.
        legacy_pads = attr.get("pads")
        if legacy_pads is not None:
            attributes.pad_pads = list(legacy_pads)
        legacy_value = attr.get("value")
        if legacy_value is not None:
            attributes.pad_constant_value = legacy_value

        # opset 11+ : pads(1), constant_value(2), axes(3) are inputs. Capture
        # values when they are constant initializers.
        if len(node.input) > 1 and node.input[1] != "":
            pads_value = attributes.get_initializer_value(model, node.input[1])
            if pads_value is not None:
                attributes.pad_pads = pads_value.tolist()
        if len(node.input) > 2 and node.input[2] != "":
            cval = attributes.get_initializer_value(model, node.input[2])
            if cval is not None:
                attributes.pad_constant_value = (
                    cval.item() if cval.size == 1 else cval.tolist()
                )
        if len(node.input) > 3 and node.input[3] != "":
            axes_value = attributes.get_initializer_value(model, node.input[3])
            if axes_value is not None:
                attributes.pad_axes = axes_value.tolist()

        # Calculating compute primitive: one write per output element.
        attributes.count_alu = np.prod(attributes.output_dimension)

        # Pad parameter inputs may be provided as initializer coefficients.
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
