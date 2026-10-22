"""Validate every code cell in a shared namespace without starting a Jupyter kernel."""

from pathlib import Path

import nbformat


def main() -> None:
    notebook = nbformat.read(Path("notebooks/privacy_risk_thresholds_colab.ipynb"), as_version=4)
    namespace = {"__name__": "__notebook_validation__"}
    for number, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        source = cell.source.replace("plt.show()", "plt.close('all')")
        code = compile(source, f"notebook-cell-{number}", "exec")
        exec(code, namespace)  # noqa: S102 - trusted, repository-owned notebook cells
    print("All notebook code cells executed successfully.")


if __name__ == "__main__":
    main()
