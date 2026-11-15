# Compression-Serve - Serve compression requests with ratio and error reporting
# Part of the Kapildhinwa2010-dev Toolkit.
"""ServeNet - model registry and request envelopes (service layer).

Lineage: the *-Serve toolkit repos. Models are registered with metadata and
version; requests travel as validated envelopes; responses carry latency and
payload size so the registry can be monitored without a web framework.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["Envelope", "ModelRecord", "ServeRegistry"]


class Envelope:
    """Validated request payload."""

    def __init__(self, request_id: str, payload: np.ndarray) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        payload = np.asarray(payload, dtype=np.float64)
        if payload.size == 0:
            raise ValueError("payload must be non-empty")
        self.request_id = request_id
        self.payload = payload

    def nbytes(self) -> int:
        return int(self.payload.nbytes)


class ModelRecord:
    """A registered model with version and callable."""

    def __init__(self, name: str, version: str, fn) -> None:
        if not name or not version:
            raise ValueError("name and version must be non-empty")
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.name = name
        self.version = version
        self.fn = fn
        self.calls = 0

    def invoke(self, payload: np.ndarray) -> np.ndarray:
        result = self.fn(payload)
        self.calls += 1
        return np.asarray(result)


class ServeRegistry:
    """Registry with versioned lookup and served-request statistics."""

    def __init__(self) -> None:
        self._models: dict = {}
        self.stats = {"requests": 0, "total_latency_ms": 0.0, "total_bytes": 0}

    def register(self, name: str, version: str, fn) -> ModelRecord:
        record = ModelRecord(name, version, fn)
        self._models[(name, version)] = record
        return record

    def get(self, name: str, version: str = "latest") -> ModelRecord:
        if version == "latest":
            candidates = [r for (n, _) in self._models if n == name]
            if not candidates:
                raise KeyError("unknown model: {}".format(name))
            version = max(v for (n, v) in self._models if n == name)
        key = (name, version)
        if key not in self._models:
            raise KeyError("unknown model: {}@{}".format(name, version))
        return self._models[key]

    def serve(self, name: str, envelope: Envelope, version: str = "latest") -> dict:
        model = self.get(name, version)
        start = time.perf_counter()
        result = model.invoke(envelope.payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats["requests"] += 1
        self.stats["total_latency_ms"] += latency_ms
        self.stats["total_bytes"] += envelope.nbytes()
        return {
            "request_id": envelope.request_id,
            "model": model.name,
            "version": model.version,
            "shape": tuple(int(d) for d in result.shape),
            "latency_ms": latency_ms,
        }

    def summary(self) -> dict:
        requests = self.stats["requests"]
        return {
            "models": len(self._models),
            "requests": requests,
            "avg_latency_ms": (self.stats["total_latency_ms"] / requests) if requests else 0.0,
            "total_bytes": self.stats["total_bytes"],
        }

__all__.append("CompressionService")


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


class CompressionService:
    """Serve compression requests; each response carries ratio and error.

    Codecs are registered callables mapping a matrix to its approximation.
    Three reference codecs ship registered by default: quantize8,
    svd_rank4 and fourier4. Compression ratio is a parameter-count
    estimate, not a measured serialized size.
    """

    def __init__(self) -> None:
        self._codecs: dict = {}
        self.stats: dict = {}
        self.register_codec("quantize8", self._quantize8)
        self.register_codec("svd_rank4", self._svd_rank4)
        self.register_codec("fourier4", self._fourier4)

    @staticmethod
    def _quantize8(matrix: np.ndarray) -> np.ndarray:
        lo, hi = float(matrix.min()), float(matrix.max())
        levels = 2 ** 8 - 1
        scale = (hi - lo) / levels if hi > lo else 1.0
        codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
        return codes * scale + lo

    @staticmethod
    def _svd_rank4(matrix: np.ndarray) -> np.ndarray:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = min(4, s.size)
        return (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    @staticmethod
    def _fourier4(matrix: np.ndarray) -> np.ndarray:
        basis = _fourier_basis(matrix.shape[1], 4)
        coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
        return coeffs.T @ basis.T

    def register_codec(self, name: str, fn) -> None:
        if not name:
            raise ValueError("codec name must be non-empty")
        if not callable(fn):
            raise TypeError("codec must be callable")
        self._codecs[name] = fn
        self.stats.setdefault(name, {"calls": 0, "total_ratio": 0.0})

    def compress(self, codec: str, matrix: np.ndarray) -> dict:
        if codec not in self._codecs:
            raise KeyError("unknown codec: {}".format(codec))
        M = np.asarray(matrix, dtype=np.float64)
        if M.ndim != 2 or M.size == 0:
            raise ValueError("matrix must be non-empty and 2-D")
        approx = np.asarray(self._codecs[codec](M), dtype=np.float64)
        norm = float(np.linalg.norm(M))
        rel_error = float(np.linalg.norm(M - approx)) / norm if norm else 0.0
        # Ratio is a parameter-count estimate (per docstring).
        m, n = M.shape
        if codec == "svd_rank4":
            k = min(4, min(m, n))
            stored_params = m * k + k + k * n
        elif codec == "fourier4":
            k = min(4, n)
            stored_params = m * k + k * n
        elif codec == "quantize8":
            stored_params = M.size  # same count, lower precision
        else:
            stored_params = approx.size
        ratio = M.size * 8 / max(stored_params * 8, 1)
        self.stats[codec]["calls"] += 1
        self.stats[codec]["total_ratio"] += ratio
        return {"codec": codec, "approx": approx,
                "relative_error": rel_error, "ratio": ratio}

    def serve(self, request_id: str, codec: str, matrix: np.ndarray) -> dict:
        """Serve one compression request; returns ratio, error and latency."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        start = time.perf_counter()
        result = self.compress(codec, matrix)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "request_id": request_id,
            "codec": result["codec"],
            "shape": tuple(int(d) for d in result["approx"].shape),
            "relative_error": result["relative_error"],
            "ratio": result["ratio"],
            "latency_ms": latency_ms,
        }

