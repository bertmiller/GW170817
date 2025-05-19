"""Tests for multi-event configuration parsing."""

from __future__ import annotations

import textwrap

import pytest

from gwsiren.config import load_config


def _base_yaml() -> str:
    return textwrap.dedent(
        """
        catalog:
          glade_plus_url: http://a
          glade24_url: http://b
          data_dir: d/
        skymap:
          default_nside: 16
          credible_level: 0.9
        mcmc:
          walkers: 4
          steps: 50
          burnin: 5
          thin_by: 1
          prior_h0_min: 1.0
          prior_h0_max: 2.0
        cosmology:
          sigma_v_pec: 100.0
          c_light: 1.0
          omega_m: 0.3
        fetcher:
          cache_dir_name: cache
          timeout: 5
          max_retries: 1
        """
    )


def test_full_multi_event_config(tmp_path):
    cfg_text = _base_yaml() + textwrap.dedent(
        """
        multi_event_analysis:
          run_settings:
            run_label: run1
            base_output_directory: out/
            candidate_galaxy_cache_dir: cache_gal/
          events_to_combine:
            - event_id: EV1
              gw_dl_samples_path: dl.npy
            - event_id: EV2
          priors:
            H0: {min: 50.0, max: 100.0}
            alpha: {min: -1.0, max: 1.0}
          mcmc:
            n_walkers: 10
            n_steps: 20
            burnin: 2
            thin_by: 1
            initial_pos_config:
              H0: {mean: 70.0, std: 5.0}
              alpha: {mean: 0.0, std: 0.2}
          cosmology:
            sigma_v_pec: 250.0
            c_light: 299792.458
            omega_m_val: 0.31
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)

    me = cfg.multi_event_analysis
    assert me is not None
    assert me.run_settings.run_label == "run1"
    assert me.run_settings.candidate_galaxy_cache_dir == "cache_gal/"
    assert len(me.events_to_combine) == 2
    assert me.priors["H0"].max == 100.0
    assert me.mcmc.n_steps == 20
    assert me.cosmology.omega_m_val == 0.31


def test_no_multi_event_section(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(_base_yaml())

    cfg = load_config(path)
    assert cfg.multi_event_analysis is None


def test_minimal_multi_event_section(tmp_path):
    cfg_text = _base_yaml() + textwrap.dedent(
        """
        multi_event_analysis:
          events_to_combine:
            - event_id: EV1
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)

    assert cfg.multi_event_analysis is not None
    assert cfg.multi_event_analysis.events_to_combine[0].event_id == "EV1"


def test_invalid_event_list(tmp_path):
    cfg_text = _base_yaml() + textwrap.dedent(
        """
        multi_event_analysis:
          events_to_combine: not_a_list
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_event_entry(tmp_path):
    cfg_text = _base_yaml() + textwrap.dedent(
        """
        multi_event_analysis:
          events_to_combine:
            - wrong: field
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    with pytest.raises(ValueError):
        load_config(path)

