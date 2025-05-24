import textwrap
import pathlib
import pytest

from gwsiren.config import load_config, _minimal_yaml_load


def test_load_config_from_path(tmp_path):
    cfg_text = textwrap.dedent(
        """
        catalog:
          glade_plus_url: http://a
          glade24_url:    http://b
          data_dir:       d/
        skymap:
          default_nside:  16
          credible_level: 0.85
        mcmc:
          walkers:        4
          steps:          50
          burnin:         5
          thin_by:        1
          prior_h0_min:   1.0
          prior_h0_max:   2.0
        cosmology:
          sigma_v_pec:    100.0
          c_light:        1.0
          omega_m:        0.3
        fetcher:
          cache_dir_name: cache
          timeout:        5
          max_retries:    1
        computation:
          backend: jax
        redshift_marginalization:
          z_err_threshold: 1e-4
          n_quad_points: 9
          sigma_range: 5.0
        mcmc_initial_positions:
          h0_mean: 72.0
          h0_std: 12.0
          alpha_mean: -0.1
          alpha_std: 0.4
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)
    assert cfg.catalog["glade_plus_url"] == "http://a"
    assert cfg.skymap["default_nside"] == 16
    assert cfg.mcmc["steps"] == 50
    assert cfg.cosmology["omega_m"] == 0.3
    assert cfg.fetcher["timeout"] == 5
    assert cfg.computation.backend == "jax"
    assert cfg.redshift_marginalization.z_err_threshold == 1e-4
    assert cfg.redshift_marginalization.n_quad_points == 9
    assert cfg.redshift_marginalization.sigma_range == 5.0
    assert cfg.mcmc_initial_positions.h0_mean == 72.0
    assert cfg.mcmc_initial_positions.h0_std == 12.0
    assert cfg.mcmc_initial_positions.alpha_mean == -0.1
    assert cfg.mcmc_initial_positions.alpha_std == 0.4


def test_load_config_missing_file(tmp_path):
    missing = tmp_path / "no_file.yaml"
    with pytest.raises(RuntimeError):
        load_config(missing)


def test_load_config_default_path(mocker, project_root_dir):
    cfg_text = textwrap.dedent(
        """
        catalog:
          glade_plus_url: http://c
          glade24_url:    http://d
          data_dir:       data/
        skymap:
          default_nside:  8
          credible_level: 0.9
        mcmc:
          walkers: 2
          steps: 10
          burnin: 1
          thin_by: 1
          prior_h0_min: 0
          prior_h0_max: 10
        cosmology:
          sigma_v_pec: 100.0
          c_light: 1.0
          omega_m: 0.3
        fetcher:
          cache_dir_name: cache
          timeout: 5
          max_retries: 2
        """
    )
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.read_text", return_value=cfg_text)

    cfg = load_config(None)

    assert cfg.catalog["glade_plus_url"] == "http://c"
    assert cfg.skymap["default_nside"] == 8
    assert cfg.mcmc["walkers"] == 2
    assert cfg.cosmology["omega_m"] == 0.3
    assert cfg.fetcher["max_retries"] == 2
    # Test that defaults are used when sections are missing
    assert cfg.computation.backend == "auto"
    assert cfg.redshift_marginalization.z_err_threshold == 1e-6
    assert cfg.mcmc_initial_positions.h0_mean == 70.0


def test_partial_new_sections(tmp_path):
    """Test that partial configuration in new sections works correctly."""
    cfg_text = textwrap.dedent(
        """
        catalog:
          glade_plus_url: http://a
          glade24_url:    http://b
          data_dir:       d/
        skymap:
          default_nside:  16
          credible_level: 0.85
        mcmc:
          walkers:        4
          steps:          50
          burnin:         5
          thin_by:        1
          prior_h0_min:   1.0
          prior_h0_max:   2.0
        cosmology:
          sigma_v_pec:    100.0
          c_light:        1.0
          omega_m:        0.3
        fetcher:
          cache_dir_name: cache
          timeout:        5
          max_retries:    1
        redshift_marginalization:
          n_quad_points: 5
        mcmc_initial_positions:
          h0_mean: 68.0
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)
    # Test that specified values are used
    assert cfg.redshift_marginalization.n_quad_points == 5
    assert cfg.mcmc_initial_positions.h0_mean == 68.0
    # Test that defaults are used for unspecified values
    assert cfg.redshift_marginalization.z_err_threshold == 1e-6  # default
    assert cfg.redshift_marginalization.sigma_range == 4.0  # default
    assert cfg.mcmc_initial_positions.h0_std == 10.0  # default
    assert cfg.mcmc_initial_positions.alpha_mean == 0.0  # default


def test_minimal_yaml_loader_valid_input():
    text = textwrap.dedent(
        """
        section1:
          key1: value1
        section2:
          key2: 2
        """
    )
    parsed = _minimal_yaml_load(text)
    assert parsed == {"section1": {"key1": "value1"}, "section2": {"key2": 2}}


def test_minimal_yaml_loader_malformed_indentation():
    malformed = textwrap.dedent(
        """
          subkey: val
        root:
          key: val
        """
    )
    with pytest.raises(ValueError):
        _minimal_yaml_load(malformed)
