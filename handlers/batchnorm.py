from node_registry import register_node_handler
from node_attributes import NodeAttributes
import numpy as np


@register_node_handler("BatchNormalization")
class BatchNormNodeHandler:
    def handle(self, model, node):
        """
        Handler for op_types "BatchNormalization"

        * Batch normalization is per-input feature map

        Args:
            model (class):  Input ONNX model
            node (class):   ONNX node

        Returns:
            attributes (class): Node attributes
        """
        attributes = NodeAttributes(model, node)

        # input_dimension is a list of per-input shapes; the first entry is the
        # data tensor (X). The scale/bias/mean/var inputs are initializers and
        # are excluded from input_dimension.
        data_shape = attributes.input_dimension[0] if attributes.input_dimension else []
        num_elements = np.prod(data_shape) if data_shape else 0
        # Channel count is the second dimension (N x C x ...); default to 1 when
        # the shape is missing or has no channel axis.
        num_channels = data_shape[1] if len(data_shape) > 1 else 1

        # Calculating compute primitive
        attributes.count_mac = num_elements
        attributes.count_alu = num_elements * 6
        attributes.count_div = num_channels * 2 + num_elements
        attributes.count_sqrt = num_channels

        # Add inputs could possibly contains coefficients
        for tensor_name in node.input:
            if attributes.is_tensor_name_initializer(model, tensor_name):
                attributes.weight_size += attributes.get_weight_size(model, tensor_name)

        return attributes
