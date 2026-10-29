"""Execute active project notebooks in fresh local kernels, including rendering.

Run: .venv/bin/python scripts/check_notebooks.py [notebook filename ...]
Executed copies and a JSON summary are saved under results/notebook_checks/.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "notebook_checks"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_PATH"] = str(ROOT / ".venv" / "share" / "jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
os.environ["JUPYTER_RUNTIME_DIR"] = str(OUTPUT / "runtime")
os.environ["IPYTHONDIR"] = str(OUTPUT / "ipython")

import nbformat
from nbclient import NotebookClient


def main():
    names = sys.argv[1:] or [p.name for p in sorted((ROOT / "notebooks").glob("*.ipynb"))]
    results = {}
    for name in names:
        path = ROOT / "notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        start = time.monotonic()
        print("Executing", name, flush=True)
        error = None
        try:
            NotebookClient(notebook, timeout=900, kernel_name="wheelchair",
                           resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
        except Exception as exc:
            error = str(exc)
        streams = "\n".join(output.get("text", "") for cell in notebook.cells
                            for output in cell.get("outputs", []) if output.output_type == "stream")
        rendering_errors = [line for line in streams.splitlines()
                            if "offscreen rendering failed" in line.lower()
                            or "offscreen rendering is unavailable" in line.lower()]
        if error is None and rendering_errors:
            error = "\n".join(rendering_errors)
        nbformat.write(notebook, OUTPUT / name)
        results[name] = {"passed": error is None, "seconds": round(time.monotonic() - start, 2),
                         "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells),
                         "image_outputs": sum("image/png" in output.get("data", {}) or "image/gif" in output.get("data", {})
                                              for cell in notebook.cells for output in cell.get("outputs", [])),
                         "error": error}
        (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print("PASS" if error is None else "FAIL", name, "" if error is None else error[-1800:], flush=True)
    return int(any(not result["passed"] for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
