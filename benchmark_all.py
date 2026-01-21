import time
import jax
import jax.numpy as jnp
import itertools
import csv
import os
from rope_ops import RoPEImplementations

# 定义日志文件名
LOG_FILENAME = "rope_benchmark_results.csv"


def verify_implementation(model, x):
    """
    验证 VPU 和 MXU 实现的输出是否一致。
    """
    print("  Verifying correctness...", end=" ")
    # 强制运行一次以编译并获取结果
    y_vpu = model.rope_vpu_standard(x).block_until_ready()
    y_mxu = model.rope_mxu_dense(x).block_until_ready()

    # print("VPU version: ", y_vpu)
    # print("MXU version: ", y_mxu)

    # 计算最大绝对误差
    # 由于浮点数计算顺序不同（VPU是逐点，MXU是累加），可能会有微小精度差异
    diff = jnp.max(jnp.abs(y_vpu - y_mxu))

    # 容忍度设为 1e-4，通常 float32 下足够判定逻辑一致性
    if diff > 1e-4:
        print(f"\n[ERROR] Mismatch detected! Max diff: {diff}")
        print("Stopping benchmark for this configuration to prevent misleading results.")
        return False
    else:
        print(f"Passed. (Max diff: {diff:.2e})")
        return True


def run_single_benchmark(batch, seq_len, heads, dim, csv_writer, file_handle):
    """
    针对一组特定的配置运行 Benchmark
    """
    config_str = f"B={batch}, L={seq_len}, H={heads}, D={dim}"
    print(f"\n>>> Running Config: {config_str}")

    try:
        # 初始化模型 (rotary_dim 设为与 head_size 相同，即 100% RoPE)
        # max_seq_len 设为当前 seq_len 的稍大值或固定值（这里设为 4096 保证足够覆盖）
        model = RoPEImplementations(head_size=dim, rotary_dim=dim, max_seq_len=max(4096, seq_len))

        # 创建随机输入
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (batch, seq_len, heads, dim))

        # 1. 正确性校验
        if not verify_implementation(model, x):
            return

        # 2. VPU Benchmark
        print("  Benchmarking VPU...", end=" ", flush=True)
        # 预热 (Warmup/Compile)
        _ = model.rope_vpu_standard(x).block_until_ready()

        # 计时循环
        start = time.time()
        loops = 100  # 循环次数
        for _ in range(loops):
            _ = model.rope_vpu_standard(x).block_until_ready()
        end = time.time()
        vpu_time_ms = (end - start) / loops * 1000
        print(f"{vpu_time_ms:.4f} ms")

        # 3. MXU Benchmark
        print("  Benchmarking MXU...", end=" ", flush=True)
        # 预热 (Warmup/Compile)
        # 注意：如果 L 很大，构建 Dense Matrix 可能会在这里 OOM
        try:
            _ = model.rope_mxu_dense(x).block_until_ready()
        except Exception as e:
            print(f"\n  [FAILED] MXU OOM or Error: {e}")
            mxu_time_ms = -1.0  # 标记失败
        else:
            # 计时循环
            start = time.time()
            for _ in range(loops):
                _ = model.rope_mxu_dense(x).block_until_ready()
            end = time.time()
            mxu_time_ms = (end - start) / loops * 1000
            print(f"{mxu_time_ms:.4f} ms")

        # 4. 记录结果
        speedup = mxu_time_ms / vpu_time_ms if mxu_time_ms > 0 else 0

        result_row = [batch, seq_len, heads, dim, f"{vpu_time_ms:.4f}", f"{mxu_time_ms:.4f}", f"{speedup:.2f}"]
        csv_writer.writerow(result_row)
        file_handle.flush()  # 立即写入磁盘防止程序崩溃丢失数据

        if mxu_time_ms > 0:
            print(f"  Result: Speedup (VPU/MXU) = {speedup:.2f}x")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed on config {config_str}: {e}")


def main():
    print(f"Running on backend: {jax.default_backend()}")
    print(f"Logging results to: {LOG_FILENAME}")

    # ================= 配置扫描范围 =================
    # 你可以在这里调整要扫描的参数组合
    # 注意：SEQ_LENS 不要设得太大（如 > 2048），因为 MXU 版本需要构建
    # [L, D, D] 的矩阵，L=4096, D=128 时矩阵约为 256MB，还在接受范围内，
    # 但如果是 L=8192 或更大，Dense MatMul 可能会直接 OOM。

    BATCH_SIZES = [1, 4, 16]
    SEQ_LENS = [128, 512, 1024, 2048]
    HEAD_COUNTS = [32]  # 通常头数对算子内部的计算密度影响是线性的
    DIMS = [64, 128]  # Head Dimension

    # ===============================================

    # 初始化 CSV 文件
    file_exists = os.path.isfile(LOG_FILENAME)

    with open(LOG_FILENAME, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 如果是新文件，写入表头
        if not file_exists:
            header = ["Batch", "Seq_Len", "Heads", "Dim", "VPU_Time_ms", "MXU_Time_ms", "Speedup_Factor"]
            writer.writerow(header)
            print(f"Created log file with header: {header}")
        else:
            print("Appending to existing log file...")

        # 生成所有组合并运行
        # itertools.product 会生成笛卡尔积，即所有参数的全排列
        combinations = list(itertools.product(BATCH_SIZES, SEQ_LENS, HEAD_COUNTS, DIMS))
        print(f"Total configurations to run: {len(combinations)}")

        for b, l, h, d in combinations:
            run_single_benchmark(b, l, h, d, writer, csvfile)

    print("\nBenchmark Suite Completed.")


if __name__ == "__main__":
    main()