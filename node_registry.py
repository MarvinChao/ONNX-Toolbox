ONNX_OPS_REGISTRY = {}


def register_node_handler(node_type):
    def wrapper(func):
        ONNX_OPS_REGISTRY[node_type] = func
        return func

    return wrapper


def get_handler(node_type):
    return ONNX_OPS_REGISTRY.get(node_type, ONNX_OPS_REGISTRY["default"])
