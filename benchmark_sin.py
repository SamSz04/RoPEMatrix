import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np

# 1. 确认设备
try:
    print(f"当前运行设备: {jax.devices()[0]}")
except:
    print("未检测到TPU/GPU，将在CPU上模拟，但原理相同。")

# 2. 准备数据
total_count = 0x0001FFFF - 0x00000000 + 1
print(f"计算总数: {total_count} (128 * 1024)")

# 创建从 0 到 131071 的浮点数
# 这里我们将整数直接作为输入值进行sin计算
flat_data = jnp.arange(total_count, dtype=jnp.float32)

# ==========================================
# 方法 A: 构造 VPU 友好的矩阵 (批量计算)
# ==========================================
# 将数据 Reshape 为 (1024, 128)。
# 128 是 TPU VPU 的典型向量宽度，这样可以确保每个时钟周期 VPU 都是满载的。
vpu_friendly_matrix = flat_data.reshape((1024, 128))

@jax.jit
def batch_sin(matrix):
    # JAX/XLA 会自动将其编译为向量化指令，一次发射处理整个矩阵
    return jnp.sin(matrix)

# ==========================================
# 方法 B: 逐个计算 (模拟设备端串行循环)
# ==========================================
# 使用 lax.scan 模拟在 TPU 上"逐个"处理。
# 虽然数据在TPU上，但我们强行让它一次只处理一个标量，
# 模拟没有利用 SIMD 并行性的情况。
@jax.jit
def sequential_sin(arr):
    def step_fn(carry, x):
        # 这里的计算是标量级别的
        return carry, jnp.sin(x)
    
    # 扫描整个数组，不利用并行化
    _, result = jax.lax.scan(step_fn, None, arr)
    return result

# ==========================================
# 性能测试
# ==========================================

print("\n--- 开始 Warmup (编译) ---")
# 第一次运行会触发 XLA 编译
_ = batch_sin(vpu_friendly_matrix).block_until_ready()
_ = sequential_sin(flat_data).block_until_ready()
print("--- 编译完成 ---\n")

# 1. 测试批量计算时间
start_time = time.time()
# 运行 100 次取平均，因为单次太快了
for _ in range(100):
    res_batch = batch_sin(vpu_friendly_matrix).block_until_ready()
end_time = time.time()
avg_batch_time = (end_time - start_time) / 100
print(f"【批量计算 (VPU Optimized)】平均耗时: {avg_batch_time * 1000:.6f} ms")

# 2. 测试逐个计算时间
# 运行 1 次 (因为慢很多)
start_time = time.time()
res_seq = sequential_sin(flat_data).block_until_ready()
end_time = time.time()
seq_time = end_time - start_time
print(f"【逐个计算 (Sequential scan)】平均耗时: {seq_time * 1000:.6f} ms")

# ==========================================
# 结果分析
# ==========================================
speedup = seq_time / avg_batch_time
print(f"\n>>> 性能差异倍数: {speedup:.2f}x")
print(f">>> 理论分析: 你的矩阵宽度为128，如果VPU完全串行，理论损失接近128倍。")
print(f">>> 实际上，由于流水线气泡和指令开销，差异通常远大于128倍。")
