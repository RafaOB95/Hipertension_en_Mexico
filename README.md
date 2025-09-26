# DS Project Template — Rafa

Organized, reproducible structure for your data science projects (inspired by Cookiecutter Data Science).

## Quick start
```bash
# 1) create and activate env (choose one)
python -m venv .venv && source .venv/bin/activate
# or: conda env create -f environment.yml && conda activate rafa-ds

# 2) install deps
pip install -r requirements.txt

# 3) configure pre-commit (format/lint on commit)
pre-commit install

# 4) run checks
make lint
make test

# 5) run pipeline example
python -m rafads.example_pipeline
```

## Repo layout
```
data/
  raw/         # nunca subas datos sensibles aquí (usa .gitignore)
  external/
  interim/
  processed/
docs/          # documentación adicional
models/        # artefactos de modelos (.pkl, .onnx) — idealmente versionados fuera de git
notebooks/     # notebooks claros y numerados: 01_..., 02_...
references/    # papers, PDFs, diccionarios de datos
reports/figures/
src/rafads/    # código importable: from rafads import ...
.github/workflows/ # CI (lint/tests)
```

## Buenas prácticas clave
- **Datos**: no subas datos crudos a Git. Usa _DVC_ o _Git LFS_ si es necesario.
- **Entorno**: congela dependencias en `requirements.txt` (o `environment.yml`).
- **Notebooks**: guarda una versión limpia sin outputs; exporta figuras a `reports/figures/`.
- **Pipelines**: la lógica va en `src/rafads/`, no dentro del notebook.
- **Trazabilidad**: usa `Makefile` para comandos repetibles.
- **Licencia**: MIT incluida; modifícala si lo necesitas.
