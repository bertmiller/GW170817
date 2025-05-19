import numpy as np
import pandas as pd

from gwsiren.h0_mcmc_analyzer import H0LogLikelihood
from gwsiren.combined_likelihood import CombinedLogLikelihood, EventDataPackage


def make_event(event_id: str, dl_values: np.ndarray, galaxy_params: dict) -> EventDataPackage:
    df = pd.DataFrame(galaxy_params)
    return EventDataPackage(event_id=event_id, dl_samples=dl_values, candidate_galaxies_df=df)


def test_prior_handling(mock_config):
    event = make_event(
        "E1",
        np.array([100.0]),
        {"z": [0.01], "mass_proxy": [1.0], "z_err": [0.0]},
    )
    cll = CombinedLogLikelihood([event], global_h0_min=50, global_h0_max=120, global_alpha_min=-1, global_alpha_max=1)

    assert cll([40.0, 0.0]) == -np.inf
    assert cll([130.0, 0.0]) == -np.inf
    assert cll([60.0, -2.0]) == -np.inf
    assert cll([60.0, 2.0]) == -np.inf


def test_single_event_matches_individual(mock_config):
    event = make_event(
        "E1",
        np.array([100.0, 110.0]),
        {"z": [0.01, 0.02], "mass_proxy": [1.0, 2.0], "z_err": [0.0, 0.0]},
    )

    cll = CombinedLogLikelihood([event])
    individual_ll = H0LogLikelihood(
        event.dl_samples,
        event.candidate_galaxies_df["z"].values,
        event.candidate_galaxies_df["mass_proxy"].values,
        event.candidate_galaxies_df["z_err"].values,
    )
    theta = [70.0, 0.0]

    assert np.isclose(cll(theta), individual_ll(theta))


def test_sum_of_multiple_events(mock_config):
    event1 = make_event(
        "E1",
        np.array([100.0]),
        {"z": [0.01], "mass_proxy": [1.0], "z_err": [0.0]},
    )
    event2 = make_event(
        "E2",
        np.array([120.0]),
        {"z": [0.015], "mass_proxy": [2.0], "z_err": [0.0]},
    )
    cll = CombinedLogLikelihood([event1, event2])

    ll1 = H0LogLikelihood(
        event1.dl_samples,
        event1.candidate_galaxies_df["z"].values,
        event1.candidate_galaxies_df["mass_proxy"].values,
        event1.candidate_galaxies_df["z_err"].values,
    )
    ll2 = H0LogLikelihood(
        event2.dl_samples,
        event2.candidate_galaxies_df["z"].values,
        event2.candidate_galaxies_df["mass_proxy"].values,
        event2.candidate_galaxies_df["z_err"].values,
    )
    theta = [70.0, 0.0]
    expected = ll1(theta) + ll2(theta)
    assert np.isclose(cll(theta), expected)


def test_event_returning_neg_inf_propagates(mock_config):
    event1 = make_event(
        "E1",
        np.array([100.0]),
        {"z": [0.01], "mass_proxy": [1.0], "z_err": [0.0]},
    )
    # negative mass_proxy will trigger -inf when alpha=1
    event2 = make_event(
        "E2",
        np.array([100.0]),
        {"z": [0.01], "mass_proxy": [-1.0], "z_err": [0.0]},
    )
    cll = CombinedLogLikelihood([event1, event2])

    assert cll([70.0, 1.0]) == -np.inf

