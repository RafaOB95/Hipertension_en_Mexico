# Migrar un proyecto de Kaggle a este repositorio

1) **Exporta** tu notebook (.ipynb) desde Kaggle.
2) **Coloca** el notebook en `notebooks/` con nombre `01_nombre_proyecto.ipynb`.
3) **Extrae lógica** repetible a módulos en `src/rafads/`.
4) **Datos**: descarga datasets a `data/raw/` (no se suben a Git). Usa variables de entorno o scripts de descarga.
5) **Resultados**: guarda figuras a `reports/figures/` y modelos a `models/`.
6) **Documenta** el flujo en el README (cómo entrenar, evaluar, reproducir).
7) **Limpia outputs** del notebook antes de hacer commit (menú: `Kernel > Restart & Clear Output`).