import textwrap
from gwsiren.config import load_config


def test_load_config(tmp_path):
    cfg_text = textwrap.dedent(
        """
        catalog:
          glade_plus_url:  http://a
          glade24_url:     http://b
          data_dir:        d/
        skymap:
          default_nside:   16
          credible_level:  0.85
        mcmc:
          walkers:         4
          steps:           50
          burnin:          5
          thin_by:         1
          prior_h0_min:    1.0
          prior_h0_max:    2.0
        cosmology:
          sigma_v_pec:     100.0
          c_light:         1.0
          omega_m:         0.3
        fetcher:
          cache_dir_name:  cache
          timeout:         5
          max_retries:     1
        computation:
          backend: "numpy"
        redshift_marginalization:
          z_err_threshold: 1e-5
          n_quad_points: 5
          sigma_range: 3.0
        mcmc_initial_positions:
          h0_mean: 65.0
          h0_std: 8.0
          alpha_mean: 0.1
          alpha_std: 0.3
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
    assert cfg.computation.backend == "numpy"
    assert cfg.redshift_marginalization.z_err_threshold == 1e-5
    assert cfg.redshift_marginalization.n_quad_points == 5
    assert cfg.redshift_marginalization.sigma_range == 3.0
    assert cfg.mcmc_initial_positions.h0_mean == 65.0
    assert cfg.mcmc_initial_positions.h0_std == 8.0
    assert cfg.mcmc_initial_positions.alpha_mean == 0.1
    assert cfg.mcmc_initial_positions.alpha_std == 0.3


def test_load_config_defaults(tmp_path):
    """Test that missing sections use defaults."""
    cfg_text = textwrap.dedent(
        """
        catalog:
          glade_plus_url:  http://a
          glade24_url:     http://b
          data_dir:        d/
        skymap:
          default_nside:   16
          credible_level:  0.85
        mcmc:
          walkers:         4
          steps:           50
          burnin:          5
          thin_by:         1
          prior_h0_min:    1.0
          prior_h0_max:    2.0
        cosmology:
          sigma_v_pec:     100.0
          c_light:         1.0
          omega_m:         0.3
        fetcher:
          cache_dir_name:  cache
          timeout:         5
          max_retries:     1
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)
    # Test defaults are used when sections are missing
    assert cfg.computation.backend == "auto"
    assert cfg.redshift_marginalization.z_err_threshold == 1e-6
    assert cfg.redshift_marginalization.n_quad_points == 7
    assert cfg.redshift_marginalization.sigma_range == 4.0
    assert cfg.mcmc_initial_positions.h0_mean == 70.0
    assert cfg.mcmc_initial_positions.h0_std == 10.0
    assert cfg.mcmc_initial_positions.alpha_mean == 0.0
    assert cfg.mcmc_initial_positions.alpha_std == 0.5
