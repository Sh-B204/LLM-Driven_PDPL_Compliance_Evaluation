#!/usr/bin/env python3
"""(Re)generate the synthetic demo DOCX files in data/demo/ (used by --mock and the tests)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdpl_eval.config import REPO_ROOT
from pdpl_eval.demo import create_demo_data

for k, v in create_demo_data(REPO_ROOT / "data" / "demo").items():
    print(k, "->", v)
