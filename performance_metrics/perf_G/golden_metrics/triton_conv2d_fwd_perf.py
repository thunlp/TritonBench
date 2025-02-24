import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.triton_conv2d_fwd import conv2d_forward
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('triton_conv2d_fwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 10):  # Example sizes, adjust as needed
            batch_size = 2 ** i
            in_channels = 3
            out_channels = 64
            height = 32
            width = 32
            kernel_size = 3
            stride = 1
            padding = 1
            input_tensor = torch.rand((batch_size, in_channels, height, width), dtype=torch.float32)
            weight_tensor = torch.rand((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32)
            self.input_tensors.append((input_tensor, weight_tensor, kernel_size, kernel_size, stride, stride, padding, padding, 1))

    def to_cuda(self, input_tensor):
        input_tensor_cuda = input_tensor[0].cuda()
        weight_tensor_cuda = input_tensor[1].cuda()
        return (input_tensor_cuda, weight_tensor_cuda) + input_tensor[2:]

    def call_op(self, input_tensor):
        return conv2d_forward(*input_tensor)

    def get_gbps(self, input_tensor, runtime):
        input_tensor_size = input_tensor[0].numel() * input_tensor[0].element_size()
        weight_tensor_size = input_tensor[1].numel() * input_tensor[1].element_size()
        output_tensor_size = self.call_op(input_tensor).numel() * input_tensor[0].element_size()
        total_bytes = input_tensor_size + weight_tensor_size + output_tensor_size
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        batch_dim, in_feat_dim, in_height, in_width = input_tensor[0].shape
        out_feat_dim, _, kernel_height, kernel_width = input_tensor[1].shape
        out_height = (in_height + 2 * input_tensor[6] - kernel_height) // input_tensor[4] + 1
        out_width = (in_width + 2 * input_tensor[7] - kernel_width) // input_tensor[5] + 1
        FLOPS = 2 * batch_dim * out_feat_dim * out_height * out_width * in_feat_dim * kernel_height * kernel_width
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
