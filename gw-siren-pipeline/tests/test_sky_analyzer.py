import numpy as np
import pandas as pd
import pytest

from gwsiren.sky_analyzer import (
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    estimate_event_specific_z_max
)


def test_generate_sky_map_and_region():
    ra = np.array([30.0, 30.2, 30.4])
    dec = np.array([-10.0, -10.1, -9.9])
    prob_map, mask, threshold = generate_sky_map_and_credible_region(ra, dec, nside=8, cdf_threshold=0.5)
    assert prob_map.sum() == pytest.approx(1.0)
    assert mask.any()
    assert 0 < threshold <= prob_map.max()


def test_select_and_filter_galaxies():
    ra = np.array([30.0, 30.2, 30.4])
    dec = np.array([-10.0, -10.1, -9.9])
    _, mask, _ = generate_sky_map_and_credible_region(ra, dec, nside=8, cdf_threshold=0.8)
    df = pd.DataFrame({
        'PGC': [1,2,3],
        'ra': [30.1, 50.0, 30.3],
        'dec': [-10.05, 0.0, -9.95],
        'z': [0.01, 0.2, 0.03]
    })
    selected = select_galaxies_in_sky_region(df, mask, nside=8)
    assert 1 <= len(selected) <= 2
    filtered = filter_galaxies_by_redshift(selected, 0.05)
    assert all(filtered['z'] < 0.05)


def test_estimate_event_specific_z_max():
    dL = np.array([40.0, 45.0, 50.0])
    z_max = estimate_event_specific_z_max(dL, percentile_dL=90.0, z_margin_factor=1.1)
    assert 0.01 <= z_max <= 0.3

