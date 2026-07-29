from src.evaluate import decide, build_comparison


def test_decide_promote_when_better_beyond_margin():
    assert decide(rmse_candidate=85.0, rmse_production=100.0, margin=0.01) == "PROMOTE"


def test_decide_keep_when_better_within_margin():
    # 99.5 < 100 pero la mejora (0.5%) no supera el margen de 1%
    assert decide(rmse_candidate=99.5, rmse_production=100.0, margin=0.01) == "KEEP"


def test_decide_keep_when_worse():
    assert decide(rmse_candidate=110.0, rmse_production=100.0, margin=0.01) == "KEEP"


def test_decide_bootstrap_when_no_production():
    assert decide(rmse_candidate=85.0, rmse_production=None, margin=0.01) == "PROMOTE"


def test_build_comparison_promote():
    c = build_comparison(rmse_candidate=85.0, rmse_production=100.0,
                         margin=0.01, git_sha="abc", evaluated_at="2026-08-03")
    assert c["recommendation"] == "PROMOTE"
    assert c["rmse_candidate"] == 85.0
    assert c["rmse_production"] == 100.0
    assert round(c["delta_pct"], 2) == -15.0     # (85-100)/100*100
    assert c["margin_pct"] == 1.0
    assert c["git_sha"] == "abc"
    assert c["evaluated_at"] == "2026-08-03"


def test_build_comparison_bootstrap():
    c = build_comparison(rmse_candidate=85.0, rmse_production=None,
                         margin=0.01, git_sha="abc", evaluated_at="2026-08-03")
    assert c["recommendation"] == "PROMOTE"
    assert c["rmse_production"] is None
    assert c["delta_pct"] is None
