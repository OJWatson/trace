import pytest


def test_lazy_import_getattr_and_dir():
    import trace

    # __dir__ should advertise submodules
    names = dir(trace)
    for name in ["analysis", "data", "model", "simulate"]:
        assert name in names

    # Accessing the attribute should lazy-import the submodule
    assert not hasattr(trace, "_nonexistent_module")
    mod = trace.data
    assert mod.__name__ == "trace.data"

    # Unknown attribute should raise AttributeError
    with pytest.raises(AttributeError):
        _ = trace.not_a_real_attr
