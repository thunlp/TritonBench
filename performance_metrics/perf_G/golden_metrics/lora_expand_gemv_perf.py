import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.lora_expand_gemv import _bgmv_expand
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('lora_expand_gemv', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 16):  # Adjust the range as needed for your tests
            batch_size = 2 ** i
            hidden_size = 64  # Example hidden size
            rank = 64  # Example rank
            lora_num = 3  # Example lora_num
            inputs = torch.rand((batch_size, hidden_size), dtype=torch.float16)
            lora_b_weights = torch.rand((lora_num, rank, hidden_size), dtype=torch.float16)
            output_tensor = torch.zeros((batch_size, hidden_size), dtype=torch.float16)
            lora_indices_tensor = torch.randint(0, 1, (batch_size,), dtype=torch.int32)
            self.input_tensors.append((inputs, lora_b_weights, output_tensor, lora_indices_tensor))

    def to_cuda(self, input_tensor):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor = input_tensor
        return (inputs.cuda(), lora_b_weights.cuda(), output_tensor.cuda(), lora_indices_tensor.cuda())

    def call_op(self, input_tensor):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor = input_tensor
        _bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs=True)
        return output_tensor

    def get_gbps(self, input_tensor, runtime):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor = input_tensor
        total_bytes = (inputs.numel() * inputs.element_size() +
                       lora_b_weights.numel() * lora_b_weights.element_size() +
                       output_tensor.numel() * output_tensor.element_size() +
                       lora_indices_tensor.numel() * lora_indices_tensor.element_size())
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        inputs, lora_b_weights, output_tensor, lora_indices_tensor = input_tensor
        FLOPS = 2 * inputs.size(0) * inputs.size(1) * lora_b_weights.size(2)  # Assuming a matrix multiplication-like operation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
