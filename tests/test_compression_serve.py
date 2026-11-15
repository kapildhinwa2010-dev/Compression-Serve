"""Tests for Compression-Serve CompressionService."""

import numpy as np
import pytest

from compression_serve import CompressionService, ServeRegistry


def _matrix():
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    return np.outer(np.sin(t), np.cos(t))


def test_default_codecs_registered():
    service = CompressionService()
    assert set(service._codecs) == {"quantize8", "svd_rank4", "fourier4"}


def test_compress_reports_error_and_ratio():
    service = CompressionService()
    result = service.compress("svd_rank4", _matrix())
    assert result["relative_error"] < 1e-6
    assert result["ratio"] > 1.0


def test_serve_returns_latency_and_shape():
    service = CompressionService()
    out = service.serve("r1", "quantize8", _matrix())
    assert out["request_id"] == "r1"
    assert out["shape"] == (32, 32)
    assert out["latency_ms"] >= 0.0


def test_unknown_codec_raises():
    service = CompressionService()
    with pytest.raises(KeyError):
        service.compress("nope", _matrix())


def test_register_custom_codec_and_stats():
    service = CompressionService()
    service.register_codec("identity", lambda m: m)
    service.compress("identity", _matrix())
    assert service.stats["identity"]["calls"] == 1


def test_compress_validates():
    service = CompressionService()
    with pytest.raises(ValueError):
        service.compress("quantize8", np.zeros(4))


def test_base_registry_still_works():
    registry = ServeRegistry()
    assert registry.summary()["requests"] == 0
