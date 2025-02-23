import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.bgmv_expand_slice import _bgmv_expand_slice
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('bgmv_expand_slice', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 12):  # Adjust the range as needed for your tests
            size = 2 ** i
            N = size
            K = size
            inputs = torch.rand((1, K), dtype=torch.float32)
            lora_b_weights = torch.rand((1, N, K), dtype=torch.float16)
            output_tensor = torch.zeros((1, N), dtype=torch.float32)
            lora_indices_tensor = torch.tensor([0], dtype=torch.int32)
            slice_offset = 0
            slice_size = N
            self.input_tensors.append((inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size))

    def to_cuda(self, input_tensor):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size = input_tensor
        return (inputs.cuda(), lora_b_weights.cuda(), output_tensor.cuda(), lora_indices_tensor.cuda(), slice_offset, slice_size)

    def call_op(self, input_tensor):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size = input_tensor
        _bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size)
        return output_tensor

    def get_gbps(self, input_tensor, runtime):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size = input_tensor
        total_bytes = inputs.numel() * inputs.element_size() + \
                      lora_b_weights.numel() * lora_b_weights.element_size() + \
                      output_tensor.numel() * output_tensor.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size = input_tensor
        N, K = lora_b_weights.shape[-2:]
        FLOPS = 2 * N * K  # Assuming a matrix-vector multiplication
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
