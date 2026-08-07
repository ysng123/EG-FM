import os
import subprocess
import sys
from pathlib import Path

import egfm


def test_public_api_and_version() -> None:
    assert egfm.__version__ == "0.1.0"
    expected = {
        "EnergyGuidedPath",
        "EnergyFlowBatch",
        "make_training_batch",
        "prediction_to_velocity",
        "evaluate_release_schedule",
        "register_release_schedule",
    }
    assert expected <= set(egfm.__all__)


def test_training_example_runs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    source_path = str((project_root / "src").resolve())
    if env.get("PYTHONPATH"):
        source_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = source_path
    result = subprocess.run(
        [sys.executable, str(project_root / "examples" / "training_step.py")],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.startswith("loss=")
