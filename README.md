# RoPEMatrix

(ropematrix) gcpuser@t1v-n-cba11816-w-0:~/RoPEMatrix$ python benchmark.py
Running on backend: tpu
Config: B=4, L=512, H=32, D=128
Initializing RoPE matrices...

--- Benchmarking VPU (Standard) ---
Compilation done.
VPU Average Time: 0.2343 ms

--- Benchmarking MXU (Dense MatMul) ---
Compilation done.
MXU Average Time: 0.1683 ms

--- Conclusion ---
Speedup (VPU / MXU): 0.72x
