import logging, math
import pytest

def test_fixtures(tmp_path, monkeypatch, caplog):
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.setattr(math, "pi", 3)
    assert math.pi == 3
    logging.getLogger("fleet_management.horizon_sweep").error("case H1=%d, H2=%d failed", 2, 5)
    assert caplog.records[0].levelname == "ERROR" and "H2=5" in caplog.text

def test_undo():
    assert math.pi > 3.1

@pytest.mark.parametrize("a, b", [(1, 2), (2, 3)])
@pytest.mark.parametrize("c", ["x", "y"])
def test_param(a, b, c):
    assert b == a + 1

def test_skip():
    pytest.importorskip("gurobipy")

def test_raises():
    with pytest.raises(ValueError, match="bad"):
        raise ValueError("bad thing")
