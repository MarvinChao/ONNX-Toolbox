import onnx
from onnx import helper, shape_inference


def detect_mish_pattern(graph_nodes):
    """
    Scans the graph for subgraphs of the form:
        X -> Softplus -> out1 -> Tanh -> out2
        (X, out2) -> Mul -> out3

    Returns a list of tuples: (mul_node, softplus_node, tanh_node)
    """
    # Build a mapping: output_name -> node
    output_map = {}
    for node in graph_nodes:
        for out in node.output:
            output_map[out] = node

    matches = []
    for node in graph_nodes:
        if node.op_type == "Mul" and len(node.input) == 2:
            in_a, in_b = node.input

            # We want one input to be X, the other input to be Tanh(Softplus(X)).
            for x_candidate, tanh_out_candidate in [(in_a, in_b), (in_b, in_a)]:
                # Check if tanh_out_candidate is produced by a Tanh node
                if tanh_out_candidate in output_map:
                    tanh_node = output_map[tanh_out_candidate]
                    if tanh_node.op_type == "Tanh" and len(tanh_node.input) == 1:
                        softplus_out = tanh_node.input[0]
                        if softplus_out in output_map:
                            softplus_node = output_map[softplus_out]
                            if (
                                softplus_node.op_type == "Softplus"
                                and len(softplus_node.input) == 1
                            ):
                                # The X that goes into Softplus should match x_candidate
                                if softplus_node.input[0] == x_candidate:
                                    matches.append((node, softplus_node, tanh_node))
    return matches


class ToolboxOptimizer:
    def __init__(self, model):
        self.model = model
        self.supported_optimizations = []
        self.register(self.fuse_mish)

    def register(self, func):
        """
        Register custom optimization functions
        """
        self.supported_optimizations.append(func)

    def apply(self):
        """
        Apply all registered optimizations in sequence
        """
        for func in self.supported_optimizations:
            func()

    def fuse_mish(self):
        """
        Fuse Softplus + Tanh operators into Mish operator
        Mish(x) = x * tanh( softplus(x) )
        """
        nodes = list(self.model.graph.node)

        # 1. Detect the subgraph pattern
        subgraph_matches = detect_mish_pattern(self.model.graph.node)
        if subgraph_matches:
            print("Softplus->Tanh->Mul pattern found.")
        else:
            return

        # 2. Perform an in-place modification while preserving order.
        # Strategy:
        #   - For each matched pattern, note the indices of the Mul, Tanh, Softplus.
        #   - We'll remove those three nodes from the node list.
        #   - Then we'll insert a new Mish node at the position where the Mul node originally was.
        #   - This ensures everything that depended on Mul's output still sees the correct topological order.

        # Convert nodes to a name->index map for quick lookups
        node_index_map = {node.name: i for i, node in enumerate(nodes)}

        # We'll collect all matches first, then fix them in ascending order of the Mul node index
        # to avoid shifting issues in the node list.
        replacements = []
        for mul_node, softplus_node, tanh_node in subgraph_matches:
            mul_idx = node_index_map[mul_node.name]
            tanh_idx = node_index_map[tanh_node.name]
            softplus_idx = node_index_map[softplus_node.name]

            # The final output of the subgraph is the Mul node’s output
            y_output = mul_node.output[0]
            # The input to the subgraph (our X) is the input to Softplus
            x_input = softplus_node.input[0]

            replacements.append(
                {
                    "mul_node": mul_node,
                    "tanh_node": tanh_node,
                    "softplus_node": softplus_node,
                    "mul_idx": mul_idx,
                    "tanh_idx": tanh_idx,
                    "softplus_idx": softplus_idx,
                    "x_input": x_input,
                    "y_output": y_output,
                }
            )

        # Sort replacements by the Mul node index (smallest first)
        replacements.sort(key=lambda r: r["mul_idx"])

        # Keep track of how many nodes we've removed so far to adjust indices
        remove_names = set()  # all node names we plan to remove

        for rep in replacements:
            remove_names.add(rep["mul_node"].name)
            remove_names.add(rep["tanh_node"].name)
            remove_names.add(rep["softplus_node"].name)

        # Build a new list of nodes; whenever we see the Mul node’s position,
        # we insert the Mish node instead (and skip the subgraph’s old nodes).
        final_nodes = []
        for i, node in enumerate(nodes):
            if node.name in remove_names:
                # We'll skip these nodes, except we'll insert the Mish node
                # at the place of the Mul node
                for rep in replacements:
                    if node.name == rep["mul_node"].name:
                        # Create the Mish node
                        mish_node = helper.make_node(
                            op_type="Mish",
                            inputs=[rep["x_input"]],
                            outputs=[rep["y_output"]],
                            name=f"Mish_{node.name}",
                            domain="",
                        )
                        final_nodes.append(mish_node)
                continue
            else:
                # If it's not one of our subgraph nodes, we keep it as is
                final_nodes.append(node)

        self.model.graph.ClearField("node")
        self.model.graph.node.extend(final_nodes)

        # 3. Shape Inference & Validation
        try:
            self.model = shape_inference.infer_shapes(self.model)
        except Exception as e:
            print("Warning: Shape inference failed:", e)

        onnx.checker.check_model(self.model)
