import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from gwsiren.config import load_config
from gwsiren.event_data import EventDataPackage

spec = importlib.util.spec_from_file_location("utils", Path(__file__).resolve().parents[1] / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
build_test_config = utils.build_test_config


@pytest.fixture
def patched_config(tmp_path, monkeypatch):
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
    return cfg


def test_execute_multi_event_analysis_flow(patched_config, mocker):
    from scripts.run_multi_event_pipeline import execute_multi_event_analysis

    packages = [
        EventDataPackage("E1", np.array([1.0]), pd.DataFrame({"z": [0.01], "mass_proxy": [1.0], "z_err": [0.001]})),
        EventDataPackage("E2", np.array([1.0]), pd.DataFrame({"z": [0.02], "mass_proxy": [1.0], "z_err": [0.001]})),
    ]
    prep_mock = mocker.patch(
        "scripts.run_multi_event_pipeline.prepare_event_data",
        side_effect=packages,
    )
    ll_instance = mocker.MagicMock()
    ll_cls = mocker.patch(
        "scripts.run_multi_event_pipeline.CombinedLogLikelihood",
        return_value=ll_instance,
    )
    sampler = mocker.MagicMock()
    run_mock = mocker.patch(
        "scripts.run_multi_event_pipeline.run_global_mcmc",
        return_value=sampler,
    )
    samples = np.ones((10, 2))
    proc_mock = mocker.patch(
        "scripts.run_multi_event_pipeline.process_global_mcmc_samples",
        return_value=samples,
    )
    save_mock = mocker.patch("scripts.run_multi_event_pipeline.save_global_samples")

    execute_multi_event_analysis()

    assert prep_mock.call_count == 2
    ll_cls.assert_called_once()
    run_mock.assert_called_once()
    proc_mock.assert_called_once_with(sampler, burnin=5, thin_by=1, n_dim=2)
    save_mock.assert_called_once()
    out_dir = Path(patched_config.multi_event_analysis.run_settings.base_output_directory)
    out_path = out_dir / patched_config.multi_event_analysis.run_settings.run_label / "global_samples.npy"
    save_mock.assert_called_with(samples, str(out_path))
