# BSc Thesis Peter Adema: Quadratic Forms in Convolutional Neural Networks

The report PDF can be [found here](report/report.pdf).
For the code used to generate it, please see:

- [report: LaTeX code and figures](report)
- [scratch: Testing and temporary files](scratch)
- [src: Model code and experiments](src)

## Experiments

The virtual environment is managed by [uv](https://docs.astral.sh/uv/getting-started/installation/),
and can be initialised with `uv sync`.
However, a working Python environment with PyTorch
and typical packages installed will likely work.

Experiments must be run with the project root as working directory, e.g.

``python src/experiments/plot_tropical_convs.py``

or, with uv synchronising the packages:

``uv run python src/experiments/plot_tropical_convs.py``