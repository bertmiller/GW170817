import builtins
import sys
import types

import pytest

from gwsiren.backends import get_xp


class DummyDevice:
    def __init__(self, platform="cpu", device_kind="CPU"):
        self.platform = platform
        self.device_kind = device_kind


def make_jax_stub(has_gpu: bool):
    jnp = types.ModuleType("jax.numpy")

    class JaxModule(types.ModuleType):
        def devices(self):
            if has_gpu:
                return [DummyDevice(platform="gpu", device_kind="GPU")]
            return [DummyDevice(platform="cpu", device_kind="CPU")]

    jax_mod = JaxModule("jax")
    jax_mod.numpy = jnp
    return jax_mod, jnp


def test_get_xp_numpy_backend(caplog):
    caplog.set_level("INFO")
    xp, name = get_xp("numpy")
    assert name == "numpy"
    assert xp is sys.modules["numpy"]
    assert any("NumPy backend" in rec.message for rec in caplog.records)


def test_get_xp_jax_not_installed(monkeypatch, caplog):
    caplog.set_level("INFO")

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("jax"):
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    xp, name = get_xp("jax")

    assert name == "numpy"
    assert xp is sys.modules["numpy"]
    assert any("not installed" in rec.message for rec in caplog.records)


def test_get_xp_auto_with_jax_cpu(monkeypatch, caplog):
    caplog.set_level("INFO")
    jax_mod, jnp_mod = make_jax_stub(has_gpu=False)
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp_mod)

    xp, name = get_xp("auto")

    assert name == "numpy"
    assert xp is sys.modules["numpy"]
    assert any("no gpu" in rec.message.lower() for rec in caplog.records)


def test_get_xp_auto_with_jax_gpu(monkeypatch, caplog):
    caplog.set_level("INFO")
    jax_mod, jnp_mod = make_jax_stub(has_gpu=True)
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp_mod)

    xp, name = get_xp("auto")

    assert name == "jax"
    assert xp is jnp_mod
    assert any("using jax on gpu" in rec.message.lower() for rec in caplog.records)
