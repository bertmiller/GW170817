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
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)
    assert cfg.catalog["glade_plus_url"] == "http://a"
    assert cfg.skymap["default_nside"] == 16
    assert cfg.mcmc["steps"] == 50
    assert cfg.cosmology["omega_m"] == 0.3
