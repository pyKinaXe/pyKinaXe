---
title: pyKinaXe
emoji: 🧬
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# pyKinaXe

Kinase extraction from PamGene phosphopeptide microarray data.
Backend for the GitHub Pages frontend.

## Runtime storage

The web backend keeps its job queue, uploaded files, logs, and generated
results under `PYKINAXE_WEB_RUNTIME_ROOT`.

- Local development: leave it unset to use `webapp/runtime/`.
- Hugging Face Spaces: attach a Storage Bucket and mount it at
  `/app/webapp/runtime` so the existing runtime folder path becomes persistent.

This lets the same queueing implementation work both locally and on Spaces
without needing a separate bucket-specific code path.

## Recommended Spaces bucket setup

For this repository's current Docker defaults:

- mount bucket `ZumbiAzul/pykinaxe` read-write at `/app/webapp/runtime`
- keep `PYKINAXE_WEB_RUNTIME_ROOT=/app/webapp/runtime`

With the current environment defaults, uploaded files and generated results are
kept in bucket-backed runtime storage and are subject to a one-hour retention
ceiling via:

- `JOB_RETENTION_HOURS=1`
- `JOB_MAX_AGE_SECONDS=3600`

Important: the web frontend also releases jobs when the user leaves the page,
and the backend reaps abandoned jobs after the heartbeat timeout. In practice,
many jobs will be deleted sooner than one hour, which is intentional to reduce
bucket growth under multi-user load.
