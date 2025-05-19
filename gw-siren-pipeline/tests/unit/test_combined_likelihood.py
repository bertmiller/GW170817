import numpy as np
import pandas as pd

from gwsiren.multi_event_data_manager import EventDataPackage
from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from gwsiren.combined_likelihood import CombinedLogLikelihood


def make_package(event_id: str, dl_value: float, z: float, mass: float) -> EventDataPackage:
    df = pd.DataFrame({"z": [z], "mass_proxy": [mass], "z_err": [0.0]})
    return EventDataPackage(event_id=event_id, dl_samples=np.array([dl_value]), candidate_galaxies_df=df)


def test_prior_handling(mock_config):
    pkg = make_package("EV", 100.0, 0.01, 1.0)
    combined = CombinedLogLikelihood([pkg])

    h0_min = combined.global_h0_min
    h0_max = combined.global_h0_max
    alpha_min = combined.global_alpha_min
    alpha_max = combined.global_alpha_max

    assert combined([h0_min - 1.0, 0.0]) == -np.inf
    assert combined([h0_max + 1.0, 0.0]) == -np.inf
    assert combined([70.0, alpha_min - 0.1]) == -np.inf
    assert combined([70.0, alpha_max + 0.1]) == -np.inf


def test_single_event_matches_individual(mock_config):
    pkg = make_package("EV1", 100.0, 0.02, 1.0)
    combined = CombinedLogLikelihood([pkg])

    individual = get_log_likelihood_h0(pkg.dl_samples, [0.02], [1.0], [0.0])
    theta = [70.0, 0.0]
    assert np.isclose(combined(theta), individual(theta))


def test_sum_of_two_events(mock_config):
    pkg1 = make_package("E1", 90.0, 0.015, 1.0)
    pkg2 = make_package("E2", 110.0, 0.025, 2.0)
    combined = CombinedLogLikelihood([pkg1, pkg2])

    ll1 = get_log_likelihood_h0(pkg1.dl_samples, [0.015], [1.0], [0.0])
    ll2 = get_log_likelihood_h0(pkg2.dl_samples, [0.025], [2.0], [0.0])

    theta = [70.0, 0.0]
    expected = ll1(theta) + ll2(theta)
    assert np.isclose(combined(theta), expected)


def test_event_returning_neg_inf_propagates(mock_config):
    good_pkg = make_package("GOOD", 100.0, 0.02, 1.0)
    bad_pkg = make_package("BAD", 100.0, 0.03, 0.0)
    combined = CombinedLogLikelihood([good_pkg, bad_pkg])

    assert combined([70.0, 1.0]) == -np.inf
