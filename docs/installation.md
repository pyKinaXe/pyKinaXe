# Installation Guide

This document describes the supported installation paths for pyKinaXe and when
to use each of them.

## Supported Python Version

- Python `3.11+`

The repository metadata in `pyproject.toml` currently targets Python 3.11 and
newer.

## Recommended Path: `environment.yml`

For most users, the simplest and most reliable installation path is:

```bash
conda env create -f environment.yml
conda activate pykinaxe
```

Why this route is recommended:

- it installs the dependencies stack in one step
- it includes Tk support for optional GUI plotting features
- it installs `inmoose` from Bioconda instead of asking pip to build it locally from source

That last point matters because pyKinaXe uses `inmoose.limma` in the peptide
statistics stage. On some machines, a direct pip installation of `inmoose`
fails if no local native-code build toolchain is available (e.g., on Windows, Microsoft Visual C++ 14.0 is required for inmoose to be installed and run properly). This installation approach installs C++ related functionality automatically.

## Alternative Path: `pip install -r requirements.txt`

If you prefer plain pip, use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the repository's full dependency list, including the web backend
dependencies.

### Important `inmoose` Caveat

PyPI currently exposes `inmoose` as a source distribution. That means:

- pip may need to compile native extensions locally
- installation can fail on systems that do not already have suitable C/C++ build tools

If that happens, install a working native build toolchain first and then rerun
pip. Typical examples are:

- macOS: Xcode Command Line Tools (https://developer.apple.com/xcode/) / Clang (https://clang.llvm.org)
- Linux: GCC/G++ and Python build headers (https://gcc.gnu.org)
- Windows: Visual Studio Build Tools / MSVC C++ (https://visualstudio.microsoft.com/visual-cpp-build-tools/)

If you do not want to manage those prerequisites yourself, go back to the
Conda route.

## Editable Package Install

For development on the Python source itself:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

This uses the package metadata from `pyproject.toml`.

## Editable Install With Web Extras

If you want the web backend dependencies from the optional `web` extra:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[web]
```

This is useful when you are developing the codebase rather than just running
the terminal pipeline.

## Which Install Path Should I Choose?

Choose `environment.yml` if:

- you want the smoothest first-time setup
- you want to avoid local compiler troubleshooting
- you want both the terminal pipeline and the web app in one ready-to-use environment

Choose `requirements.txt` if:

- you already manage your own Python environments
- you are comfortable troubleshooting compiler/toolchain issues if `inmoose` needs a local build

Choose `pip install -e .` if:

- you are developing the package itself
- you want edits in `src/` and `config/` to apply immediately without reinstalling

## Local Run Commands After Installation

Terminal pipeline:

```bash
python scripts/kx_kinase_extraction_pipeline.py
```

Local web backend:

```bash
python webapp/pykinaxe_webapp.py
```

## Hugging Face / Web Runtime Note

The web app stores queue state, uploads, logs, and generated artifacts under
`PYKINAXE_WEB_RUNTIME_ROOT`.

- locally, you can leave that unset and use `webapp/runtime/`
- in Hugging Face Spaces, point it at a mounted Storage Bucket path

This means the same codepath works both locally and on Hugging Face without a
separate storage adapter.

## Dependency Sources Referenced In This Repository

- `requirements.txt`: pip-oriented full dependency list
- `environment.yml`: conda-oriented environment, including Bioconda `inmoose`
- `pyproject.toml`: package metadata and editable-install dependency surface
- `setup.py`: minimal setuptools shim that defers to `pyproject.toml`
