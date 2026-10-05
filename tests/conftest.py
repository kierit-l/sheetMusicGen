import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_hand_model(request, monkeypatch):
    """Fast tests split hands at middle C instead of loading the hand-part model (a download)."""
    if request.node.get_closest_marker("slow"):
        return
    from sheetmusicgen import pm2s

    monkeypatch.setattr(pm2s, "right_hand_probs", lambda notes: np.array([float(n.pitch >= 60) for n in notes]))


@pytest.fixture(autouse=True)
def no_score_library(monkeypatch, tmp_path):
    """Tests don't see a score library installed on this machine (each test may build its own)."""
    from sheetmusicgen import lookup

    monkeypatch.setattr(lookup, "INDEX_DIR", tmp_path / "no_scores")
