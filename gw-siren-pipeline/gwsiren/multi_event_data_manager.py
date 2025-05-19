"""Utilities for preparing data from multiple GW events.

This module provides helpers to load or generate per-event data needed for a
combined analysis. Candidate galaxy lists are cached to avoid repeatedly
processing large galaxy catalogues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import yaml

from gwsiren.pipeline import (
    run_full_analysis,
    NSIDE_SKYMAP,
    CDF_THRESHOLD,
    CATALOG_TYPE,
    HOST_Z_MAX_FALLBACK,
)
from gwsiren import CONFIG
from gwsiren.gw_data_fetcher import configure_astropy_cache, fetch_candidate_data
from gwsiren.event_data_extractor import extract_gw_event_parameters

from gwsiren.event_data import EventDataPackage

logger = logging.getLogger(__name__)

def load_multi_event_config(path: str | Path) -> List[Dict]:
    """Load the multi-event configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        List of dictionaries, one per event configuration entry.

    Raises:
        ValueError: If required structure is missing.
    """
    cfg_path = Path(path)
    text = cfg_path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "events_to_combine" not in raw:
        raise ValueError("Config file must define 'events_to_combine'.")
    events = raw["events_to_combine"]
    if not isinstance(events, list):
        raise ValueError("'events_to_combine' must be a list.")
    for entry in events:
        if not isinstance(entry, dict) or "event_id" not in entry:
            raise ValueError("Each event entry must contain 'event_id'.")
    return events


def _fetch_dl_samples(event_id: str) -> np.ndarray:
    """Fetch luminosity distance samples for a GW event."""
    cache_dir = configure_astropy_cache(CONFIG.fetcher["cache_dir_name"])
    if not cache_dir:
        raise RuntimeError("Failed to configure astropy cache")
    success, gw_data_obj = fetch_candidate_data(event_id, cache_dir)
    if not success:
        raise RuntimeError(f"Failed to fetch GW data: {gw_data_obj}")
    dl_samples, _, _ = extract_gw_event_parameters(gw_data_obj, event_id)
    if dl_samples is None:
        raise RuntimeError("Essential GW parameters are missing")
    return dl_samples


def prepare_event_data(event_cfg_entry: Dict) -> EventDataPackage:
    """Prepare data for a single event.

    This loads pre-computed data if paths are provided and valid; otherwise it
    triggers generation via :func:`run_full_analysis`.

    Args:
        event_cfg_entry: Configuration dictionary for the event.

    Returns:
        ``EventDataPackage`` with loaded or generated data.

    Raises:
        RuntimeError: If data generation fails.
    """
    event_id = event_cfg_entry["event_id"]

    raw_proc_params = event_cfg_entry.get("single_event_processing_params")
    proc_params = raw_proc_params if raw_proc_params is not None else {}
    nside = proc_params.get("nside_skymap", event_cfg_entry.get("nside_skymap", NSIDE_SKYMAP))
    cdf = proc_params.get("cdf_threshold", event_cfg_entry.get("cdf_threshold", CDF_THRESHOLD))
    catalog = proc_params.get("catalog_type", event_cfg_entry.get("catalog_type", CATALOG_TYPE))
    z_fallback = proc_params.get(
        "host_z_max_fallback", event_cfg_entry.get("host_z_max_fallback", HOST_Z_MAX_FALLBACK)
    )

    cache_dir = "cache/candidate_galaxies"
    me_cfg = getattr(CONFIG, "multi_event_analysis", None)
    if me_cfg and me_cfg.run_settings and me_cfg.run_settings.candidate_galaxy_cache_dir:
        cache_dir = me_cfg.run_settings.candidate_galaxy_cache_dir
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cat_key = catalog.replace("/", "_").replace(" ", "_")
    cache_file = cache_path / f"{event_id}_cat_{cat_key}_n{nside}_cdf{cdf}_zfb{z_fallback}.csv"

    dl_samples = None
    dl_path = event_cfg_entry.get("gw_dl_samples_path")
    if dl_path:
        dl_file = Path(dl_path)
        if dl_file.exists():
            dl_samples = np.load(dl_file)
        else:
            logger.warning("gw_dl_samples_path not found for %s", event_id)

    if cache_file.exists():
        logger.info("Loading cached candidate galaxies for %s from %s", event_id, cache_file)
        candidate_df = pd.read_csv(cache_file)
        if dl_samples is None:
            dl_samples = _fetch_dl_samples(event_id)
        return EventDataPackage(event_id=event_id, dl_samples=dl_samples, candidate_galaxies_df=candidate_df)

    logger.info("Generating data for %s", event_id)
    results = run_full_analysis(
        event_id,
        perform_mcmc=False,
        nside_skymap=nside,
        cdf_threshold=cdf,
        catalog_type=catalog,
        host_z_max_fallback=z_fallback,
    )
    if results.get("error"):
        raise RuntimeError(f"Data generation failed for {event_id}: {results['error']}")

    dl_samples = results["dL_samples"]
    candidate_df = results["candidate_hosts_df"]

    try:
        candidate_df.to_csv(cache_file, index=False)
        logger.info("Saved candidate galaxies for %s to %s", event_id, cache_file)
    except Exception as exc:  # pragma: no cover - unexpected I/O errors
        logger.warning("Could not save candidate galaxies for %s: %s", event_id, exc)

    return EventDataPackage(event_id=event_id, dl_samples=dl_samples, candidate_galaxies_df=candidate_df)


def prepare_all_event_data(multi_event_config_path: str | Path) -> List[EventDataPackage]:
    """Prepare data packages for all events defined in a configuration file."""
    events = load_multi_event_config(multi_event_config_path)
    packages = []
    for entry in events:
        packages.append(prepare_event_data(entry))
    return packages


__all__ = [
    "EventDataPackage",
    "load_multi_event_config",
    "prepare_event_data",
    "prepare_all_event_data",
]
