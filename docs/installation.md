# Installation

pyKinaXe requires Python 3.11 or newer (see `pyproject.toml`).

## Conda (recommended)

```bash
conda env create -f environment.yml
conda activate pykinaxe
```

This installs the whole dependency stack in one step, including the web
backend, and takes `inmoose` from Bioconda as a prebuilt package. pyKinaXe uses
`inmoose.limma` in the peptide statistics stage; a pip installation of
`inmoose` has to compile native code and fails on machines without a C/C++
toolchain (on Windows, Microsoft Visual C++ 14.0 or newer). Plotting is
headless (matplotlib Agg), so no Tk is needed.

## pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` lists the full dependency set, including the web backend.
PyPI provides `inmoose` as a source distribution, so pip may need to compile
it. If that fails, install a build toolchain first and rerun pip:

- macOS: Xcode Command Line Tools (https://developer.apple.com/xcode/)
- Linux: GCC/G++ and the Python development headers (https://gcc.gnu.org)
- Windows: Visual Studio Build Tools with the C++ workload
  (https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Editable install

For development on the package itself, so that edits in `src/` and `config/`
apply without reinstalling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .          # pipeline only
pip install -e .[web]     # with the Flask/gunicorn web backend
```

The package metadata is in `pyproject.toml`; `setup.py` is a shim for tools
that still expect it.

## Running

Terminal pipeline:

```bash
python scripts/kx_kinase_extraction_pipeline.py
```

Local web backend:

```bash
python webapp/pykinaxe_webapp.py
```

The web app stores its queue state, uploads, logs and results under
`PYKINAXE_WEB_RUNTIME_ROOT`. Leave it unset locally to use `webapp/runtime/`;
on Hugging Face Spaces point it at the mounted Storage Bucket.
