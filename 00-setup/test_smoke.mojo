# 00-setup / test_smoke.mojo
#
# Pure-CPU sanity tests. Confirms `mojo test` works without GPU access —
# useful in CI and on the x86_64 Linux box where the RTX 4090 isn't attached.

from testing import assert_equal, assert_almost_equal


fn simd_sum[width: Int](xs: SIMD[DType.float32, width]) -> Float32:
    return xs.reduce_add()


def test_simd_sum_width_4():
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    assert_almost_equal(simd_sum[4](v), 10.0)


def test_simd_sum_width_8():
    var v = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    assert_almost_equal(simd_sum[8](v), 36.0)


def test_int_arithmetic():
    # Sanity: confirm Int operations and the test runner are wired.
    assert_equal(2 + 2, 4)
    assert_equal(10 // 3, 3)
