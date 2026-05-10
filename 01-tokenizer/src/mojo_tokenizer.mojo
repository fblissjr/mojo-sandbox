# 01-tokenizer / mojo_tokenizer.mojo
#
# Phase A: minimal Mojo function exported as a Python module.
# Validates the build + import + call pipeline before layering BPE complexity.
#
# Build:  mojo build --emit shared-lib src/mojo_tokenizer.mojo -o build/mojo_tokenizer.so
# Use:    python -c "import sys; sys.path.insert(0, 'build'); import mojo_tokenizer; print(mojo_tokenizer.count_bytes('hello'))"

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


def count_bytes(text: PythonObject) -> PythonObject:
    """Return the byte length of `text` (str), computed in Mojo."""
    var s = String(text)
    return PythonObject(s.byte_length())


@export
def PyInit_mojo_tokenizer() -> PythonObject:
    """Create the mojo_tokenizer Python module."""
    try:
        var b = PythonModuleBuilder("mojo_tokenizer")
        b.def_function[count_bytes](
            "count_bytes",
            docstring="Return len() of a Python str, computed in Mojo.",
        )
        return b.finalize()
    except e:
        abort(t"failed to create mojo_tokenizer module: {e}")
