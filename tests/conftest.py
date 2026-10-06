import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from pdpl_eval.config import load_config
from pdpl_eval.demo import create_demo_data


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    create_demo_data(d)
    return d


@pytest.fixture()
def cfg(tmp_path, demo_dir):
    """Demo config with data and results redirected to temp directories."""
    c = load_config("configs/demo/experiment.yaml", root=ROOT)
    c.retrieval["corpus_path"] = str(demo_dir / "DemoPDPL.docx")
    from pdpl_eval.config import Application
    c.applications = [Application("DemoShort", "demo_short", demo_dir / "DemoShort.docx"),
                      Application("DemoLong", "demo_long", demo_dir / "DemoLong.docx")]
    return c.with_results_dir(tmp_path / "results")
