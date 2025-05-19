from gwsiren.config import MEPriorBoundaries, MECosmologyConfig
from scripts.run_multi_event_pipeline import _resolve_prior, _resolve_value


def test_resolve_prior_returns_override():
    priors = {"H0": MEPriorBoundaries(min=50.0, max=100.0)}
    assert _resolve_prior(priors, "H0", 20.0, 150.0) == (50.0, 100.0)


def test_resolve_prior_defaults():
    assert _resolve_prior(None, "H0", 20.0, 150.0) == (20.0, 150.0)


def test_resolve_value_uses_attr():
    obj = MECosmologyConfig(sigma_v_pec=250.0)
    assert _resolve_value(obj, "sigma_v_pec", 200.0) == 250.0


def test_resolve_value_fallback():
    obj = MECosmologyConfig()
    assert _resolve_value(obj, "sigma_v_pec", 200.0) == 200.0
