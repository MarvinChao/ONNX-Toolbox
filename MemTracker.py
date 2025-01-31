from copy import deepcopy


def is_model_output(tensor_name, model):
    """
    Check if a tensor is a final output of the model.
    """
    for output in model.graph.output:
        if output.name == tensor_name:
            return True
    return False


def is_model_input(tensor_name, model):
    """
    Check if a tensor is an input of the model.
    """
    for input in model.graph.input:
        if input.name == tensor_name:
            return True
    return False


class MemTracker:
    """
    This class is used to track memory traffic and footprint in the scenario of using local SRAM.
    The implementation here is meant to be naive for common sense practices. Users of specific HW/SW environment
    can override this class with more aggressive optimizationss techniques

    Attributes:
    model (class):              Input ONNX model
    tensor_size (dict):         Tensor size map for the model
    ref_count (dict):           This track the tensor that is being referenced in the ONNX model
    local_memory_size (int):    The size of local memory, used to determine buffer eviction policy
    in_local_memory (set):      Track the tensor that resides in local memory
    current_footprint (int):    The local mempory usage
    max_foorprint (int):        The maximum memory usage
    bytes_loaded_total (int):   Number of bytes loaded from system dram
    bytes_stored_total (int):   Number of bytes stored to system memory
    """

    def __init__(self, model, tensor_size, ref_count, local_memory_size):
        self.model = model
        self.tensor_size = tensor_size
        self.ref_count = deepcopy(ref_count)
        self.local_memory_size = local_memory_size
        self.in_local_memory = set()
        self.current_footprint = 0
        self.max_footprint = 0

        self.bytes_loaded_total = 0
        self.bytes_stored_total = 0

    def process_node(self, node, next_node_chainable):
        """
        Simulate memory management with local SRAM

        Returns updated (bytes_loaded, bytes_stored, current_foorprint, max_footprint, chainable),
        """
        # Track local loads/stores for just this node
        bytes_loaded_node = 0
        bytes_stored_node = 0

        # 1. Load inputs if not already in local SRAM
        for input in node.input:
            if input not in self.in_local_memory:
                load_size = self.tensor_size.get(input, 0)
                bytes_loaded_node += load_size
                self.bytes_loaded_total += load_size
                self.in_local_memory.add(input)
                self.current_footprint += load_size
                if self.current_footprint > self.max_footprint:
                    self.max_footprint = self.current_footprint

        # 2. Allocate outputs in local SRAM
        for output in node.output:
            if output:
                out_size = self.tensor_size.get(output, 0)
                self.in_local_memory.add(output)
                self.current_footprint += out_size
                if self.current_footprint > self.max_footprint:
                    self.max_footprint = self.current_footprint

        # 3. If the next node is "un-chainable", flush its outputs to DRAM.
        if not next_node_chainable:
            # Flush all of this node's outputs
            for output in node.output:
                if output in self.in_local_memory:
                    out_size = self.tensor_size.get(output, 0)
                    bytes_stored_node += out_size
                    self.bytes_stored_total += out_size

                    # free from local memory
                    self.in_local_memory.remove(output)
                    self.current_footprint -= out_size

        # 4. Free inputs with ref_count=0
        for input in node.input:
            if input in self.ref_count:
                self.ref_count[input] -= 1
                if self.ref_count[input] == 0:
                    # If it's a final model output, flush it.
                    if is_model_output(input, self.model):
                        store_size = self.tensor_size.get(input, 0)
                        bytes_stored_node += store_size
                        self.bytes_stored_total += store_size

                    # remove from SRAM
                    if input in self.in_local_memory:
                        input_size = self.tensor_size.get(input, 0)
                        self.current_footprint -= input_size
                        self.in_local_memory.remove(input)

        # Return the per-node metrics
        return {
            "bytes_loaded": bytes_loaded_node,
            "bytes_stored": bytes_stored_node,
            "footprint": self.current_footprint,
            "max_footprint": self.max_footprint,
        }

    def finalize(self):
        """
        After all nodes are processed, we store any leftover final outputs
        that remain in SRAM. Returns a dict summarizing that final store
        plus final usage stats.
        """
        bytes_stored_final = 0

        # Store all final graph outputs that remain
        for output in self.model.graph.output:
            output_name = output.name
            if output_name in self.in_local_memory:
                store_size = self.tensor_size.get(output_name, 0)
                bytes_stored_final += store_size
                self.bytes_stored_total += store_size

                self.current_footprint -= store_size
                self.in_local_memory.remove(output_name)

        return {
            "bytes_stored_final": bytes_stored_final,
            "final_footprint": self.current_footprint,
            "max_footprint": self.max_footprint,
            "total_bytes_loaded": self.bytes_loaded_total,
            "total_bytes_stored": self.bytes_stored_total,
        }
