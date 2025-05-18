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
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_text)

    cfg = load_config(path)
    assert cfg.catalog["glade_plus_url"] == "http://a"
    assert cfg.skymap["default_nside"] == 16
    assert cfg.mcmc["steps"] == 50
    assert cfg.cosmology["omega_m"] == 0.3


def test_load_config_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(RuntimeError):
        load_config(missing)


def test_load_config_default_path(mocker, project_root_dir):
    cfg_text = textwrap.dedent(
        """
        catalog:
          glade_plus_url: http://x
          glade24_url:    http://y
          data_dir:       d/
        skymap:
          default_nside:  8
          credible_level: 0.5
        mcmc:
          walkers:        2
          steps:          10
          burnin:         1
          thin_by:        1
          prior_h0_min:   1.0
          prior_h0_max:   2.0
        cosmology:
          sigma_v_pec:    100.0
          c_light:        1.0
          omega_m:        0.3
        """
    )

    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.read_text", return_value=cfg_text)

    cfg = load_config()
    assert cfg.catalog["glade_plus_url"] == "http://x"
    assert cfg.skymap["default_nside"] == 8
    assert cfg.mcmc["steps"] == 10
    assert cfg.cosmology["omega_m"] == 0.3


def test_minimal_yaml_loader_valid_input():
    text = "root:\n  a: 1\nother:\n  b: 2"
    data = _minimal_yaml_load(text)
    assert data == {"root": {"a": 1}, "other": {"b": 2}}


def test_minimal_yaml_loader_malformed_indentation():
    malformed = "  bad: 1\nroot:\n  a: 2"
    with pytest.raises(ValueError):
        _minimal_yaml_load(malformed)
