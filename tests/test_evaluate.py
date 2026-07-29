from src.evaluate import decide


def test_decide_promote_when_better_beyond_margin():
    assert decide(rmse_candidate=85.0, rmse_production=100.0, margin=0.01) == "PROMOTE"


def test_decide_keep_when_better_within_margin():
    # 99.5 < 100 pero la mejora (0.5%) no supera el margen de 1%
    assert decide(rmse_candidate=99.5, rmse_production=100.0, margin=0.01) == "KEEP"


def test_decide_keep_when_worse():
    assert decide(rmse_candidate=110.0, rmse_production=100.0, margin=0.01) == "KEEP"


def test_decide_bootstrap_when_no_production():
    assert decide(rmse_candidate=85.0, rmse_production=None, margin=0.01) == "PROMOTE"
