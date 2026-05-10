# 00-setup / hello_kernel.mojo
#
# End-to-end GPU smoke test: vector add on the local accelerator.
# Targets Mojo 1.0 beta (Modular 26.3, dev nightly).
#
# Kernel-arg pattern: LayoutTensor[..., MutAnyOrigin] (raw UnsafePointer
# isn't DevicePassable in 1.0). Buffers are wrapped in a Tensor before launch.

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from layout import Layout, LayoutTensor


comptime N = 1 << 20            # 1,048,576 elements (~4 MB per fp32 buffer)
comptime BLOCK = 256
comptime GRID = (N + BLOCK - 1) // BLOCK
comptime layout = Layout.row_major(N)
comptime Tensor = LayoutTensor[DType.float32, layout, MutAnyOrigin]


def vec_add_kernel(a: Tensor, b: Tensor, output: Tensor):
    var i = global_idx.x
    if i < N:
        output[i] = a[i] + b[i]


def main() raises:
    comptime if not has_accelerator():
        print("No GPU detected. Requires Metal (Apple) or CUDA (NVIDIA).")
        return

    var ctx = DeviceContext()
    print("Detected accelerator. Launching vec_add: N=", N, " block=", BLOCK, " grid=", GRID)

    var dev_a = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_b = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_out = ctx.enqueue_create_buffer[DType.float32](N)

    with dev_a.map_to_host() as a_host:
        for i in range(N):
            a_host[i] = Float32(i)
    with dev_b.map_to_host() as b_host:
        for i in range(N):
            b_host[i] = Float32(i) * 2.0

    var a_t = Tensor(dev_a)
    var b_t = Tensor(dev_b)
    var out_t = Tensor(dev_out)

    ctx.enqueue_function[vec_add_kernel](
        a_t, b_t, out_t,
        grid_dim=GRID,
        block_dim=BLOCK,
    )
    ctx.synchronize()

    var passed = True
    with dev_out.map_to_host() as out_host:
        for i in range(3):
            print("out[", i, "] = ", out_host[i])
        for i in range(N):
            var expected = Float32(i) + Float32(i) * 2.0
            var diff = out_host[i] - expected
            if diff < -1.0e-5 or diff > 1.0e-5:
                passed = False
                print("MISMATCH at i=", i, " got=", out_host[i], " expected=", expected)
                break

    if passed:
        print("PASS: vector add matches reference on ", N, " elements")
    else:
        print("FAIL: see mismatch above")
