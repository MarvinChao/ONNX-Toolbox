"""
Helpers to build small, deterministic ONNX graphs for regression testing.

Everything here is hermetic: no models are downloaded and no randomness is
used, so tests are fully reproducible. The builders keep explicit value_info /
input / output shapes so the toolbox handlers (which read shapes from the
graph) have everything they need without relying on shape inference.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper


def make_tensor_value_info(name, shape, elem_type=TensorProto.FLOAT):
    """
    Create a typed, shaped tensor value info.

    Args:
        name (str):         Tensor name.
        shape (list[int]):  Tensor dimensions.
        elem_type (int):    ONNX TensorProto element type (default FLOAT).

    Returns:
        onnx.ValueInfoProto
    """
    return helper.make_tensor_value_info(name, elem_type, shape)


def make_initializer(name, array, elem_type=TensorProto.FLOAT):
    """
    Create an initializer tensor from a numpy array or a python list.

    Args:
        name (str):                     Tensor name.
        array (np.ndarray | list):      Values for the initializer.
        elem_type (int):                ONNX TensorProto element type.

    Returns:
        onnx.TensorProto
    """
    np_dtype = helper.tensor_dtype_to_np_dtype(elem_type)
    np_array = np.asarray(array, dtype=np_dtype)
    return helper.make_tensor(
        name=name,
        data_type=elem_type,
        dims=list(np_array.shape),
        vals=np_array.flatten().tolist(),
    )


def make_single_node_model(
    op_type,
    inputs,
    outputs,
    input_infos,
    output_infos,
    initializers=None,
    value_infos=None,
    attributes=None,
    opset=13,
):
    """
    Build a valid single-node ONNX model.

    Args:
        op_type (str):              The ONNX op_type (e.g. "Conv").
        inputs (list[str]):         Node input tensor names (order matters).
        outputs (list[str]):        Node output tensor names.
        input_infos (list):         ValueInfoProto for graph inputs.
        output_infos (list):        ValueInfoProto for graph outputs.
        initializers (list):        Optional initializer TensorProtos.
        value_infos (list):         Optional intermediate value infos.
        attributes (dict):          Optional node attributes.
        opset (int):                Opset version for the model.

    Returns:
        onnx.ModelProto
    """
    node = helper.make_node(
        op_type,
        inputs=inputs,
        outputs=outputs,
        **(attributes or {}),
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"{op_type}_test_graph",
        inputs=input_infos,
        outputs=output_infos,
        initializer=initializers or [],
        value_info=value_infos or [],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    return model, node


def make_sequential_model(node_specs, graph_input_info, graph_output_name, opset=13):
    """
    Build a linear (single-path) multi-node ONNX model for end-to-end tests.

    Each spec describes one node whose single input is the previous node's
    output (or the graph input for the first node).

    Args:
        node_specs (list[dict]): Each dict has:
            - op_type (str)
            - output_info (ValueInfoProto): shape/type of this node's output
            - initializers (list, optional)
            - extra_inputs (list[str], optional): additional input names after
              the primary data input
            - attributes (dict, optional)
        graph_input_info (ValueInfoProto): the model input.
        graph_output_name (str):          name of the final output tensor.
        opset (int):                      opset version.

    Returns:
        onnx.ModelProto
    """
    nodes = []
    initializers = []
    value_infos = []

    prev_output = graph_input_info.name
    last_index = len(node_specs) - 1

    for i, spec in enumerate(node_specs):
        out_info = spec["output_info"]
        out_name = graph_output_name if i == last_index else out_info.name

        node_inputs = [prev_output] + list(spec.get("extra_inputs", []))
        node = helper.make_node(
            spec["op_type"],
            inputs=node_inputs,
            outputs=[out_name],
            **(spec.get("attributes", {})),
        )
        nodes.append(node)
        initializers.extend(spec.get("initializers", []))

        # Intermediate outputs are declared as value_info so handlers can read
        # their shapes; the final output is a graph output.
        if i != last_index:
            value_infos.append(
                make_tensor_value_info(out_name, _shape_of(out_info), _type_of(out_info))
            )
        prev_output = out_name

    final_info = make_tensor_value_info(
        graph_output_name,
        _shape_of(node_specs[last_index]["output_info"]),
        _type_of(node_specs[last_index]["output_info"]),
    )

    graph = helper.make_graph(
        nodes=nodes,
        name="sequential_test_graph",
        inputs=[graph_input_info],
        outputs=[final_info],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model


def _shape_of(value_info):
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _type_of(value_info):
    return value_info.type.tensor_type.elem_type
