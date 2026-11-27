"""Read-only verification that protected research files are unchanged."""

import csv
import hashlib

from utils.data_loader import REPO_ROOT


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_28_protected_hashes_unchanged():
    with (REPO_ROOT / "docs" / "app_source_hashes.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 28
    mismatches = [r["File"] for r in rows if _sha256(REPO_ROOT / r["File"]) != r["SHA256"].upper()]
    assert mismatches == []


def test_final_test_lock_present():
    assert (REPO_ROOT / "reports" / "final_test.lock").exists()


def test_runtime_does_not_import_training_libraries():
    """Render every page in a fresh interpreter and check no training library loads."""
    import subprocess
    import sys

    script = f"""
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(r"{REPO_ROOT / 'app' / 'streamlit_app.py'}", default_timeout=240)
at.run()
for page in ["pages/explore.py", "pages/priority.py", "pages/historical.py", "pages/model_insights.py"]:
    at.switch_page(page); at.run()
    assert not at.exception, (page, [e.value for e in at.exception])
bad = [m for m in ("lightgbm", "shap", "torch", "sklearn", "xgboost") if m in sys.modules]
print("FORBIDDEN:", bad)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert "FORBIDDEN: []" in result.stdout, result.stdout + result.stderr
