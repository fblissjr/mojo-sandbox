# 00-setup / test_smoke.mojo
#
# Runnable as `mojo run test_smoke.mojo` (Mojo 1.0 dropped `mojo test`).
# CPU-only sanity tests. Confirms compile + run pipeline without GPU access.

from std.testing import assert_equal, assert_almost_equal


def test_simd_sum_width_4() raises:
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    assert_almost_equal(v.reduce_add(), Float32(10.0))


def test_simd_sum_width_8() raises:
    var v = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    assert_almost_equal(v.reduce_add(), Float32(36.0))


def test_int_arithmetic() raises:
    assert_equal(2 + 2, 4)
    assert_equal(10 // 3, 3)


def main() raises:
    test_simd_sum_width_4()
    test_simd_sum_width_8()
    test_int_arithmetic()
    print("OK: 3 smoke tests passed")
