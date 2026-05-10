# 01-tokenizer / mojo_tokenizer.mojo
#
# Phase A: byte-length passthrough — proves the Python extension pipeline.
# Phase B: SIMD-chunked byte counter with comptime width specialization.
#
# Build:  mojo build --emit shared-lib src/mojo_tokenizer.mojo -o build/mojo_tokenizer.so

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


# ---------- Phase A --------------------------------------------------------

def count_bytes(text: PythonObject) -> PythonObject:
    """Return the byte length of `text` (str), computed in Mojo."""
    var s = String(text)
    return PythonObject(s.byte_length())


# ---------- Phase B --------------------------------------------------------

def count_byte_scalar(text: PythonObject, target: PythonObject) raises -> PythonObject:
    """Scalar baseline: count `target` bytes via plain loop. Bench reference."""
    var s = String(text)
    var t = UInt8(Int(py=target))
    var n = s.byte_length()
    var ptr = s.unsafe_ptr()

    var count: Int = 0
    var i: Int = 0
    while i < n:
        if ptr[i] == t:
            count += 1
        i += 1
    return PythonObject(count)


def count_byte_simd[width: Int](text: PythonObject, target: PythonObject) raises -> PythonObject:
    """Count `target` bytes via comptime-`width` SIMD chunked scan."""
    var s = String(text)
    var t = UInt8(Int(py=target))
    var n = s.byte_length()
    var ptr = s.unsafe_ptr()

    var t_vec = SIMD[DType.uint8, width](t)
    var acc = SIMD[DType.uint64, 1](0)
    var i: Int = 0

    # Vectorized body: load `width` bytes, elementwise eq vs broadcast target
    # (SIMD == SIMD returns a scalar Bool in 1.0; .eq() gives the lanewise mask),
    # cast bool lanes to uint8 (0/1), reduce_add into a u64 accumulator.
    while i + width <= n:
        var v = ptr.load[width=width](i)
        acc += SIMD[DType.uint64, 1](Int(v.eq(t_vec).cast[DType.uint8]().reduce_add()))
        i += width

    # Scalar tail for the remainder.
    while i < n:
        if ptr[i] == t:
            acc += SIMD[DType.uint64, 1](1)
        i += 1

    return PythonObject(Int(acc[0]))


# ---------- Python module entry point --------------------------------------

@export
def PyInit_mojo_tokenizer() -> PythonObject:
    """Create the mojo_tokenizer Python module."""
    try:
        var b = PythonModuleBuilder("mojo_tokenizer")

        # Phase A
        b.def_function[count_bytes](
            "count_bytes",
            docstring="Return len() of a Python str, computed in Mojo.",
        )

        # Phase B — scalar baseline
        b.def_function[count_byte_scalar](
            "count_byte_scalar",
            docstring="Count occurrences of byte `target` in `text` (scalar loop).",
        )

        # Phase B — SIMD specializations: each is a separately compiled
        # instantiation of the same comptime-parameterized function.
        b.def_function[count_byte_simd[16]](
            "count_byte_simd_16",
            docstring="SIMD-16 byte counter.",
        )
        b.def_function[count_byte_simd[32]](
            "count_byte_simd_32",
            docstring="SIMD-32 byte counter.",
        )
        b.def_function[count_byte_simd[64]](
            "count_byte_simd_64",
            docstring="SIMD-64 byte counter.",
        )

        return b.finalize()
    except e:
        abort(t"failed to create mojo_tokenizer module: {e}")
