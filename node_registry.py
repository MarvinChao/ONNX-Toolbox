ONNX_OPS_REGISTRY = {}


def register_node_handler(node_type):
    def wrapper(cls):
        ONNX_OPS_REGISTRY[node_type] = cls
        return cls

    return wrapper


def get_handler(node_type):
    handler = ONNX_OPS_REGISTRY.get(node_type, ONNX_OPS_REGISTRY["default"])
    return handler()
