import importlib.util
import sys
from pathlib import Path


def load_example(name):
    path = Path(__file__).parents[2] / "example" / "fvm" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ns_simple_example_accepts_linear_solver_argument(monkeypatch):
    example = load_example("ns_fvm_simple_example.py")
    captured = {}

    class FakeModel:
        def __init__(self, options):
            captured.update(options)

        def __str__(self):
            return "FakeModel"

        def solve(self, **kwargs):
            return None

        def compute_error(self):
            return 0.0, 0.0, 0.0

    monkeypatch.setattr(example, "NSFVMSimpleModel", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ns_fvm_simple_example.py",
            "--backend",
            "numpy",
            "--linear_solver",
            "scipy",
            "--max_iter",
            "1",
            "--no-pbar_log",
        ],
    )

    example.main()

    assert captured["linear_solver_config"].solver == "scipy"


def test_stokes_simple_example_accepts_linear_solver_argument(monkeypatch):
    example = load_example("stokes_fvm_simple_example.py")
    captured = {}

    class FakeModel:
        def __init__(self, options):
            captured.update(options)

        def __str__(self):
            return "FakeModel"

        def solve(self, **kwargs):
            return None

        def compute_error(self):
            return 0.0, 0.0, 0.0

    monkeypatch.setattr(example, "StokesFVMSimpleModel", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stokes_fvm_simple_example.py",
            "--backend",
            "numpy",
            "--linear_solver",
            "mumps",
            "--max_iter",
            "1",
            "--no-pbar_log",
        ],
    )

    example.main()

    assert captured["linear_solver_config"].solver == "mumps"
