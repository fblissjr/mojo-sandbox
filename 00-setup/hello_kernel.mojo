# 00-setup / hello_kernel.mojo
#
# End-to-end GPU smoke test: vector add on the local accelerator.
# Targets Mojo 1.0 beta. If APIs in the gpu.host / gpu modules have shifted,
# check docs.modular.com/mojo/manual/gpu/intro-tutorial/ — the names below
# match the public examples as of Modular 26.3.

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from sys import has_accelerator
from memory import UnsafePointer
from math import abs as fabs


fn vec_add_kernel(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    n: Int,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        out[i] = a[i] + b[i]


def main():
    @parameter
    if not has_accelerator():
        print("No GPU detected. This file requires Metal (Apple) or CUDA (NVIDIA).")
        print("Confirm with `mojo --version` and check the runtime install.")
        return

    alias N = 1 << 20            # 1,048,576 elements (~4 MB per buffer at fp32)
    alias BLOCK = 256
    alias GRID = (N + BLOCK - 1) // BLOCK

    var ctx = DeviceContext()
    print("Detected accelerator. Launching vec_add: N=", N, " block=", BLOCK, " grid=", GRID)

    # Allocate device-resident buffers.
    var dev_a = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_b = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_out = ctx.enqueue_create_buffer[DType.float32](N)

    # Fill inputs from the host. enqueue_create_host_buffer + copy is the
    # standard pattern; on Apple unified memory it's a near-noop, on CUDA it's
    # a real PCIe copy.
    with dev_a.map_to_host() as a_host:
        for i in range(N):
            a_host[i] = Float32(i)
    with dev_b.map_to_host() as b_host:
        for i in range(N):
            b_host[i] = Float32(i) * 2.0

    ctx.enqueue_function[vec_add_kernel](
        dev_a.unsafe_ptr(),
        dev_b.unsafe_ptr(),
        dev_out.unsafe_ptr(),
        N,
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
            if fabs(out_host[i] - expected) > 1.0e-5:
                passed = False
                print("MISMATCH at i=", i, " got=", out_host[i], " expected=", expected)
                break

    if passed:
        print("PASS: vector add matches reference on ", N, " elements")
    else:
        print("FAIL: see mismatch above")
