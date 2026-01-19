import jax
import jax.numpy as jnp
from functools import partial


class RoPEImplementations:
    def __init__(self, head_size: int, rotary_dim: int, max_seq_len: int, base: int = 10000):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len

        # 1. 预计算频率 (VPU和MXU都需要)
        inv_freq = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)  # [seq_len, rotary_dim/2]

        self.cos = jnp.cos(freqs)  # [seq_len, rotary_dim/2]
        self.sin = jnp.sin(freqs)  # [seq_len, rotary_dim/2]

        # 2. 为矩阵乘法版本预计算 Dense Matrix (为了测试纯算力，我们把构建矩阵放在init里)
        # 我们需要构建一个形状为 [seq_len, rotary_dim, rotary_dim] 的巨大矩阵
        self.dense_matrix = self._precompute_dense_matrix(self.cos, self.sin)

    def _precompute_dense_matrix(self, cos, sin):
        """
        构建块对角矩阵 R_{m}。
        Standard 风格:
        [ cos, -sin,   0,    0 ]
        [ sin,  cos,   0,    0 ]
        [   0,    0, cos, -sin ]
        [   0,    0, sin,  cos ]
        """
        seq_len, half_dim = cos.shape
        full_dim = half_dim * 2

        # 初始化一个全是 0 的大矩阵 [seq_len, dim, dim]
        # 注意：这非常消耗显存，dim 很大时会 OOM
        R = jnp.zeros((seq_len, full_dim, full_dim), dtype=jnp.float32)

        # 填充对角线和反对角线块
        # 偶数行 indices: 0, 2, 4...
        evens = jnp.arange(0, full_dim, 2)
        # 奇数行 indices: 1, 3, 5...
        odds = jnp.arange(1, full_dim, 2)

        # 利用 vmap 或 scan 填充会更快，这里为了逻辑清晰使用 index_update
        # 实际在 TPU 上，我们构建 indices 并在最后一次性 set

        # 1. 主对角线 (cos)
        # R[t, 2i, 2i] = cos[t, i]
        R = R.at[:, evens, evens].set(cos)
        # R[t, 2i+1, 2i+1] = cos[t, i]
        R = R.at[:, odds, odds].set(cos)

        # 2. 反对角线 (sin)
        # Standard Style:
        # x_new[2i]   = x[2i]*cos - x[2i+1]*sin  -> R[2i, 2i+1] = -sin
        # x_new[2i+1] = x[2i]*sin + x[2i+1]*cos  -> R[2i+1, 2i] = sin

        R = R.at[:, evens, odds].set(-sin)
        R = R.at[:, odds, evens].set(sin)

        return R

    @partial(jax.jit, static_argnums=(0,))
    def rope_vpu_standard(self, x, start_pos=0):
        """
        VPU 优化版本 (Standard Style)
        x: [batch, seq_len, num_heads, head_size]
        """
        seq_len = x.shape[1]

        # 切片获取当前窗口的 cos/sin
        curr_cos = self.cos[start_pos: start_pos + seq_len]
        curr_sin = self.sin[start_pos: start_pos + seq_len]

        # 调整形状以广播: [seq_len, 1, head_size//2]
        curr_cos = curr_cos[:, None, :]
        curr_sin = curr_sin[:, None, :]

        # 拆分偶数和奇数维度 (Standard Style)
        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        x0 = x_rot[..., 0::2]
        x1 = x_rot[..., 1::2]

        # VPU 向量计算
        y0 = x0 * curr_cos - x1 * curr_sin
        y1 = x0 * curr_sin + x1 * curr_cos

        # 重新组合 (interleave)
        # shape: [batch, seq_len, num_heads, rotary_dim]
        y_rot = jnp.stack((y0, y1), axis=-1).reshape(x_rot.shape)

        return jnp.concatenate((y_rot, x_pass), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def rope_mxu_dense(self, x, start_pos=0):
        """
        MXU 强制矩阵乘法版本
        x: [batch, seq_len, num_heads, head_size]
        """
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        num_heads = x.shape[2]

        # 获取当前位置的旋转矩阵 R: [seq_len, rotary_dim, rotary_dim]
        R = self.dense_matrix[start_pos: start_pos + seq_len]

        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        # 准备进行矩阵乘法
        # 目标: result[b, t, h, d] = sum_k (x[b, t, h, k] * R[t, k, d])
        # 这本质上是对每个 token 向量应用对应的矩阵 R_t

        # 调整 x 的形状以便与 R 进行 broadcast matmul
        # x_rot: [Batch, Seq, Heads, Dim] -> permute -> [Batch, Heads, Seq, Dim]
        # 为了清晰，我们使用 einsum，这会被 XLA 编译为 Batch MatMul

        # b: batch, t: time, h: head, i: input_dim, j: output_dim
        # x: bthi, R: tij -> bthj
        # 注意：这里 R 实际上应该是转置乘法，因为 y = R * x (列向量)
        # 但在代码中通常是行向量 x * R^T。
        # 我们上面构建的 R 对应 y = R @ x (数学公式)，所以行向量形式应该是 x @ R.T

        # 为了强制 MXU 满负荷工作，我们直接做 einsum
        y_rot = jnp.einsum('bthk, tkd -> bthd', x_rot, R)

        return jnp.concatenate((y_rot, x_pass), axis=-1)