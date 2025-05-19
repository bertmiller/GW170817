import numpy as np
import matplotlib
matplotlib.use("Agg")

from gwsiren.plot_utils import (
    load_combined_samples,
    load_individual_event_samples,
    get_event_ids_from_config,
    plot_overlaid_1d_posteriors,
    plot_combined_corner,
)
from gwsiren.config import MultiEventAnalysisSettings, MEEventToCombine


def test_load_combined_samples(tmp_path):
    data = np.random.rand(5, 2)
    f = tmp_path / "comb.npy"
    np.save(f, data)
    loaded = load_combined_samples(str(f))
    assert np.allclose(loaded, data)


def test_load_individual_event_samples(tmp_path):
    d = tmp_path / "H0_samples_E1.npy"
    arr = np.array([1.0, 2.0])
    np.save(d, arr)
    loaded = load_individual_event_samples("E1", str(tmp_path))
    assert np.allclose(loaded, arr)
    missing = load_individual_event_samples("NOPE", str(tmp_path))
    assert missing is None


def test_get_event_ids_from_config():
    cfg = MultiEventAnalysisSettings(events_to_combine=[MEEventToCombine(event_id="A"), MEEventToCombine(event_id="B")])
    ids = get_event_ids_from_config(cfg)
    assert ids == ["A", "B"]


def test_plot_functions(tmp_path):
    indiv = {"E": np.random.normal(70, 5, 50)}
    combined = np.column_stack((np.random.normal(70, 2, 100), np.random.normal(0, 1, 100)))

    over_path = tmp_path / "over.pdf"
    plot_overlaid_1d_posteriors(indiv, combined, "H0", str(over_path))
    assert over_path.exists()

    corner_path = tmp_path / "corner.pdf"
    plot_combined_corner(combined, str(corner_path))
    assert corner_path.exists()
