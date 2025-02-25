import torch
import triton
import triton.language as tl

from typing import Callable
import json
import os

class do_bench_config():
    def __init__(
            self,
            warm_up=25,
            repetition=100,
            grad_to_none=None,
            quantiles=[0.5, 0.8, 0.2],
            return_mode="median"
    ):
        self.warm_up = warm_up
        self.repetition = repetition
        self.grad_to_none = grad_to_none
        self.quantiles = quantiles
        self.return_mode = return_mode

class Performance_Metrics:
    def __init__(
            self,
            op_name,
            dtype=None,
            is_backward=False,
            **kwargs
    ):
        self.op_name = op_name # 算子名称
        self.dtype = dtype # 算子输入类型
        if is_backward:
            self.op_name += 'backward'
        self.kwargs = kwargs

        self.input_tensors = []
        self.do_bench_config = do_bench_config()

    """
    获取测试算子性能所需输入张量
    """
    def get_input_tensors(self):
        raise NotImplementedError("You must implement this method to get input tensors")

    """
    将输入张量转移到cuda
    """
    def to_cuda(self, input_tensor):
        raise NotImplementedError("You must implement this method to get input tensors")
    
    """
    调用算子
    """
    def call_op(self, input_tensor):
        raise NotImplementedError("You must implement this method to call the op")

    """
    获取能够使算子运行时间稳定下来的设置
    """
    def get_do_bench_config(self, warmup=None, rep=None):
        if warmup != None and rep != None:
            self.do_bench_config = do_bench_config(
                warm_up=warmup,
                repetition=rep,
            )
            return

        if self.input_tensors == []: # 必须提供输入输出
            raise NotImplementedError("You must implement this method to get input_tensors")
        
        previous_ms = None
        epsilon = 1e-4  # 容忍度，判断ms是否稳定
        stable_count = 0  # 记录稳定的次数
        max_stable_count = 3  # 允许的稳定次数，超过则认为 ms 已经稳定
        input_tensor = self.to_cuda(self.input_tensors[-1])

        for t in range(1, 11):
            # 设置 warmup 和 rep 参数
            warmup = 100 * t
            rep = 1000 * t
            
            # 进行基准测试
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: self.call_op(input_tensor),
                warmup=warmup,
                rep=rep,
                quantiles=[0.5, 0.8, 0.2],
                return_mode="median"
            )

            print("warmup time:", warmup, "rep time:", rep, "runtime:", ms)

            # 如果有前一个 ms 值，检查是否稳定
            if previous_ms is not None:
                # 计算相对变化
                relative_change = abs(ms - previous_ms) / abs(previous_ms) if previous_ms != 0 else float('inf')

                # 判断相对变化是否小于 epsilon
                if relative_change < epsilon:
                    stable_count += 1
                else:
                    stable_count = 0
            
            # 如果连续稳定达到 max_stable_count 次，认为 ms 稳定
            if stable_count >= max_stable_count:
                print(f"MS stabilized with warmup={warmup} and rep={rep}")
                self.do_bench_config = do_bench_config(
                    warm_up=warmup,
                    repetition=rep,
                )
                return

            # 更新 previous_ms 为当前 ms
            previous_ms = ms
        
        # 如果没有找到稳定的 ms，返回默认配置
        print("MS did not stabilize. Returning default config.")
        raise NotImplementedError("You must implement this method to make the runtime stable")

    """
    获取算子运行一次的绝对时间
    """
    def get_runtime(self, op: Callable):
        ms, min_ms, max_ms = triton.testing.do_bench(
            op,
            warmup=self.do_bench_config.warm_up,
            rep=self.do_bench_config.repetition,
            quantiles=self.do_bench_config.quantiles,
            return_mode=self.do_bench_config.return_mode
        )
        return ms
    
    """
    获取算子运行一次的GBPS
    """
    def get_gbps(self, input_tensor, runtime):
        raise NotImplementedError("You must implement this method to get the method to calculate GBPS")

    """
    获取算子运行一次的TFLOPS
    """
    def get_tflops(self, input_tensor, runtime):
        raise NotImplementedError("You must implement this method to get the method to calculate TFLOPS")

    """
    对所有的输入，测试运行时的算子性能
    """
    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
            try:
                input_tensor = self.to_cuda(input_tensor_)
                # print(input_tensor)
                op = lambda : self.call_op(input_tensor)            
                ms = self.get_runtime(op)
                gbps = self.get_gbps(input_tensor, ms)
                tflops = self.get_tflops(input_tensor, ms)
                result = {
                    "input_size": [item.shape if type(item)==torch.Tensor else item for item in input_tensor],
                    "ms": ms,
                    "GB/s": gbps,
                    "TFLOPS": tflops
                }
                print(result)
                results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark for input tensor. Error: {e}")
            input_tensor = None
        folder_path = "./r1_0_shot_results"
        file_name = self.op_name + ".json"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)
