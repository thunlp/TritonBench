import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.sgmv_expand_slice import _sgmv_expand_slice
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('sgmv_expand_slice', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 12):  # Adjust the range as needed for your testing
            size = 2 ** i
            inputs = torch.rand(size, size, dtype=torch.float32)
            lora_b_weights = torch.rand(1024, size, size, dtype=torch.float16)
            output_tensor = torch.zeros_like(inputs)
            b_seq_start_loc = torch.zeros(size, dtype=torch.int32)
            seq_len_tensor = torch.full((size,), size, dtype=torch.int32)
            lora_indices_tensor = torch.zeros(size, dtype=torch.int32)
            self.input_tensors.append((inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor, lora_indices_tensor))

    def to_cuda(self, input_tensor):
        return tuple(tensor.cuda() for tensor in input_tensor)

    def call_op(self, input_tensor):
        inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor, lora_indices_tensor = input_tensor
        _sgmv_expand_slice(
            inputs,
            lora_b_weights,
            output_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches=inputs.size(0),
            max_seq_length=inputs.size(0),
            token_nums=inputs.size(0),
            slice_offset=0,
            slice_size=lora_b_weights.size(-2),
            add_inputs=False
        )
        return output_tensor

    def get_gbps(self, input_tensor, runtime):
        inputs, lora_b_weights, output_tensor, _, _, _ = input_tensor
        total_bytes = inputs.numel() * inputs.element_size() + lora_b_weights.numel() * lora_b_weights.element_size() + output_tensor.numel() * output_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        inputs, lora_b_weights, _, _, _, _ = input_tensor
        FLOPS = 2 * inputs.size(0) * inputs.size(1) * lora_b_weights.size(-1)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
