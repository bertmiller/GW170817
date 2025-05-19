import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

from gwsiren.config import load_config
spec = importlib.util.spec_from_file_location("utils", Path(__file__).resolve().parents[1] / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
build_test_config = utils.build_test_config

from gwsiren.event_data import EventDataPackage



def test_gaussian_combination_validation(tmp_path, monkeypatch, mocker):
    cfg_path = build_test_config(tmp_path, ["E1", "E2"])
    cfg = load_config(cfg_path)
    targets = [
        "gwsiren.config.CONFIG",
        "gwsiren.multi_event_data_manager.CONFIG",
        "gwsiren.combined_likelihood.CONFIG",
        "gwsiren.global_mcmc.CONFIG",
        "scripts.run_multi_event_pipeline.CONFIG",
    ]
    for t in targets:
        monkeypatch.setattr(t, cfg, raising=False)

    from scripts.run_multi_event_pipeline import execute_multi_event_analysis

    packages = [
        EventDataPackage("E1", np.array([1.0]), pd.DataFrame({"z": [0.01], "mass_proxy": [1], "z_err": [0.001]})),
        EventDataPackage("E2", np.array([1.0]), pd.DataFrame({"z": [0.02], "mass_proxy": [1], "z_err": [0.001]})),
    ]
    mocker.patch("scripts.run_multi_event_pipeline.prepare_event_data", side_effect=packages)

    mean1, mean2 = 70.0, 74.0
    sigma = 5.0
    expected_mean = (mean1 + mean2) / 2
    expected_sigma = sigma / np.sqrt(2)

    class MockLL:
        def __init__(self, _pkgs, **kwargs):
            self.h0_min = kwargs.get("global_h0_min")
            self.h0_max = kwargs.get("global_h0_max")
            self.alpha_min = kwargs.get("global_alpha_min")
            self.alpha_max = kwargs.get("global_alpha_max")

        def __call__(self, theta):
            h0, alpha = theta
            if not (self.h0_min <= h0 <= self.h0_max):
                return -np.inf
            if not (self.alpha_min <= alpha <= self.alpha_max):
                return -np.inf
            return -0.5 * (((h0 - mean1) / sigma) ** 2 + ((h0 - mean2) / sigma) ** 2)

    monkeypatch.setattr("scripts.run_multi_event_pipeline.CombinedLogLikelihood", MockLL)

    def fake_run_global_mcmc(ll, n_walkers, n_steps, n_dim, initial_pos_config, pool):
        rng = np.random.default_rng(0)
        chain = rng.normal(expected_mean, expected_sigma, size=(n_steps, n_walkers))
        chain2 = np.stack((chain, np.zeros_like(chain)), axis=2)
        sampler = mocker.MagicMock()
        sampler.get_chain.return_value = chain2
        return sampler

    monkeypatch.setattr("scripts.run_multi_event_pipeline.run_global_mcmc", fake_run_global_mcmc)

    def fake_process(sampler, burnin, thin_by, n_dim):
        chain = sampler.get_chain()[burnin::thin_by]
        return chain.reshape(-1, n_dim)

    monkeypatch.setattr("scripts.run_multi_event_pipeline.process_global_mcmc_samples", fake_process)
    monkeypatch.setattr("scripts.run_multi_event_pipeline.save_global_samples", lambda samples, out: np.save(out, samples))

    execute_multi_event_analysis()

    out_file = Path(cfg.multi_event_analysis.run_settings.base_output_directory) / cfg.multi_event_analysis.run_settings.run_label / "global_samples.npy"
    samples = np.load(out_file)
    h0_samples = samples[:, 0]
    assert abs(h0_samples.mean() - expected_mean) < 1.0
    assert abs(h0_samples.std(ddof=1) - expected_sigma) < 0.5
