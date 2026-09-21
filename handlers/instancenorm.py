from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("InstanceNormalization")
class InstanceNormNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "InstanceNormalization"

        * Layer normalization is per-batch, per-channel

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # input_dimension is a list of per-input shapes; the first entry is the
        # data tensor (X). The scale/bias inputs are initializers and are
        # excluded from input_dimension.
        data_shape = attributes.input_dimension[0] if attributes.input_dimension else []
        num_elements = np.prod(data_shape) if data_shape else 0
        # Normalization is per-batch, per-channel (N x C x ...).
        num_batch = data_shape[0] if len(data_shape) > 0 else 1
        num_channels = data_shape[1] if len(data_shape) > 1 else 1
        batch_channel = num_batch * num_channels

        # Calculating compute primitive
        attributes.count_mac = num_elements
        attributes.count_alu = num_elements * 6
        attributes.count_div = batch_channel * 2 + num_elements
        attributes.count_sqrt = batch_channel

        # Add inputs could possibly contains coefficients
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
