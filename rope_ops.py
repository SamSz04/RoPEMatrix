import jax
import jax.numpy as jnp
from functools import partial


class RoPEImplementations:
    def __init__(self, head_size: int, rotary_dim: int, max_seq_len: int, base: int = 10000):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len

        # 1. 预计算频率
        inv_freq = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)  # [seq_len, rotary_dim/2]

        self.cos = jnp.cos(freqs)  # [seq_len, rotary_dim/2]
        self.sin = jnp.sin(freqs)  # [seq_len, rotary_dim/2]

        # 2. 预计算 Dense Matrix
        self.dense_matrix = self._precompute_dense_matrix(self.cos, self.sin)

    def _precompute_dense_matrix(self, cos, sin):
        """
        构建用于右乘的矩阵 R (Right-multiplication Matrix).

        目标公式 (Standard Style):
        y0 = x0*cos - x1*sin
        y1 = x0*sin + x1*cos

        如果我们执行 y = x @ R (x是行向量 [x0, x1]), 那么 R 必须是:
        [ cos,  sin ]  <- 第一行对应 x0 的贡献 (R[0,0] -> y0, R[0,1] -> y1)
        [ -sin, cos ]  <- 第二行对应 x1 的贡献 (R[1,0] -> y0, R[1,1] -> y1)

        验证:
        [x0, x1] @ [ [cos, sin], [-sin, cos] ] = [x0*cos - x1*sin,  x0*sin + x1*cos] -> 正确!
        """
        seq_len, half_dim = cos.shape
        full_dim = half_dim * 2

        # 初始化 [seq_len, dim, dim]
        R = jnp.zeros((seq_len, full_dim, full_dim), dtype=jnp.float32)

        evens = jnp.arange(0, full_dim, 2)  # 0, 2, 4...
        odds = jnp.arange(1, full_dim, 2)  # 1, 3, 5...

        # 1. 主对角线 (cos)
        R = R.at[:, evens, evens].set(cos)
        R = R.at[:, odds, odds].set(cos)

        # 2. 反对角线 (sin) - 这里是修正的关键点！
        # 我们需要 R[0, 1] (偶行奇列) = sin
        # 我们需要 R[1, 0] (奇行偶列) = -sin

        # 之前的错误代码: R.at[:, evens, odds].set(-sin)
        # 修正后的代码:
        R = R.at[:, evens, odds].set(sin)  # 第一行第二列：贡献给 y1 的 x0 部分 (+sin)
        R = R.at[:, odds, evens].set(-sin)  # 第二行第一列：贡献给 y0 的 x1 部分 (-sin)

        return R

    @partial(jax.jit, static_argnums=(0,))
    def rope_vpu_standard(self, x, start_pos=0):
        """VPU 优化版本 (Standard Style)"""
        seq_len = x.shape[1]

        curr_cos = self.cos[start_pos: start_pos + seq_len]
        curr_sin = self.sin[start_pos: start_pos + seq_len]

        curr_cos = curr_cos[:, None, :]
        curr_sin = curr_sin[:, None, :]

        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        x0 = x_rot[..., 0::2]
        x1 = x_rot[..., 1::2]

        # Standard RoPE formula:
        # y0 = x0 * cos - x1 * sin
        # y1 = x0 * sin + x1 * cos
        y0 = x0 * curr_cos - x1 * curr_sin
        y1 = x0 * curr_sin + x1 * curr_cos

        y_rot = jnp.stack((y0, y1), axis=-1).reshape(x_rot.shape)

        return jnp.concatenate((y_rot, x_pass), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def rope_mxu_dense(self, x, start_pos=0):
        """MXU 强制矩阵乘法版本"""
        seq_len = x.shape[1]

        # R: [seq_len, rotary_dim, rotary_dim]
        R = self.dense_matrix[start_pos: start_pos + seq_len]

        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        # einsum '...k, ...kd -> ...d' 等价于 MatMul x @ R
        # 这里 R 已经是针对右乘优化过的了
        y_rot = jnp.einsum('bthk, tkd -> bthd', x_rot, R)

        return jnp.concatenate((y_rot, x_pass), axis=-1)