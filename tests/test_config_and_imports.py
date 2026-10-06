import importlib
import pkgutil
import pytest

import pdpl_eval
from pdpl_eval.config import ConfigError, load_config, select
from conftest import ROOT


def test_all_modules_import():
    for m in pkgutil.iter_modules(pdpl_eval.__path__):
        importlib.import_module(f"pdpl_eval.{m.name}")


def test_real_config_loads():
    c = load_config("configs/experiment.yaml", root=ROOT)
    assert len(c.applications) == 19 and [m.id for m in c.models] == ["allam-7b", "llama-3.1-8b", "qwen-2.5-7b", "mistral-7b"]
    assert c.runs == 5 and set(c.strategies) == {"section_rag", "full_rag", "section_norag", "full_norag"}
    sel = select(c)
    assert len(sel.strategies) == 4 and sel.run_numbers == [1, 2, 3, 4, 5]
    assert len(sel.applications) * len(sel.models) * len(sel.strategies) * 5 == 19 * 4 * 4 * 5
    for s in c.strategies.values():
        assert s.prompt_path.exists()


def test_filters_and_errors():
    c = load_config("configs/experiment.yaml", root=ROOT)
    sel = select(c, ["circlys", "Dinar"], ["qwen-2.5-7b"], ["full_rag"], runs=2)
    assert [a.id for a in sel.applications] == ["circlys", "dinar"] and sel.run_numbers == [1, 2]
    with pytest.raises(ConfigError):
        select(c, ["nope"])
    with pytest.raises(ConfigError):
        select(c, strategies=["bogus"])
