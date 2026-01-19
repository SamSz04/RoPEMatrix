import time
import jax.numpy as jnp
import jax
from rope_ops import RoPEImplementations


def benchmark():
    print(f"Running on backend: {jax.default_backend()}")

    # 配置参数 (模拟 Llama-2-7B 的配置)
    BATCH = 4
    SEQ_LEN = 512  # 序列不宜过长，否则 Dense Matrix 会爆显存
    HEADS = 32
    DIM = 128  # head_size
    ROT_DIM = 128  # 全旋转

    print(f"Config: B={BATCH}, L={SEQ_LEN}, H={HEADS}, D={DIM}")

    # 初始化模型
    print("Initializing RoPE matrices...")
    model = RoPEImplementations(DIM, ROT_DIM, 2048)

    # 创建 Dummy Input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (BATCH, SEQ_LEN, HEADS, DIM))

    # 1. 编译并预热 VPU 版本
    print("\n--- Benchmarking VPU (Standard) ---")
    _ = model.rope_vpu_standard(x).block_until_ready()  # 触发编译
    print("Compilation done.")

    start = time.time()
    for _ in range(100):
        _ = model.rope_vpu_standard(x).block_until_ready()
    end = time.time()
    vpu_time = (end - start) / 100 * 1000
    print(f"VPU Average Time: {vpu_time:.4f} ms")

    # 2. 编译并预热 MXU 版本
    print("\n--- Benchmarking MXU (Dense MatMul) ---")
    _ = model.rope_mxu_dense(x).block_until_ready()  # 触发编译
    print("Compilation done.")

    start = time.time()
    for _ in range(100):
        _ = model.rope_mxu_dense(x).block_until_ready()
    end = time.time()
    mxu_time = (end - start) / 100 * 1000
    print(f"MXU Average Time: {mxu_time:.4f} ms")

    print("\n--- Conclusion ---")
    print(f"Speedup (VPU / MXU): {mxu_time / vpu_time:.2f}x")
    print("Note: The MXU version performs massive amounts of multiplication by zero.")


if __name__ == "__main__":
    benchmark()