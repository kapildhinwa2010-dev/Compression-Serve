# Compression-Serve

[![CI](https://github.com/kapildhinwa2010-dev/Compression-Serve/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Compression-Serve/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Compression-Serve/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Compression-Serve/actions/workflows/codeql.yml)

Serve compression requests: registered codecs return ratio and reconstruction error per call.

## What it does

CompressionService registers codecs (callables mapping a matrix to its
approximation) and serves compression requests. Three reference codecs
ship by default - quantize8, svd_rank4, fourier4 - and every response
carries relative reconstruction error, an estimated compression ratio
and latency. Per-codec call statistics accumulate.

    from compression_serve import CompressionService

    service = CompressionService()
    out = service.serve("req-1", "svd_rank4", W)
    print(out["ratio"], out["relative_error"], out["latency_ms"])

## API

- CompressionService.register_codec / compress / serve / stats
- Default codecs: quantize8, svd_rank4, fourier4
- ServeRegistry - the generic base registry (unchanged from DNA-Serve)

## Honest results and limitations

- Compression ratio is a parameter-count estimate (payload bytes over
  approximation bytes), not a measured serialized file size.
- Relative error is Frobenius norm against the input matrix only; nothing
  here predicts downstream model accuracy.
- Single-threaded, in-process; no queueing or concurrency model.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
