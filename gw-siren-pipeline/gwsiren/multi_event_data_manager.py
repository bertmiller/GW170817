"""Utilities for preparing data from multiple GW events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


def _load_precomputed_data(entry: Dict) -> EventDataPackage | None:
    dl_path = entry.get("gw_dl_samples_path")
    gal_path = entry.get("candidate_galaxies_path")
    if not dl_path or not gal_path:
        return None
    dl_file = Path(dl_path)
    gal_file = Path(gal_path)
    if not (dl_file.exists() and gal_file.exists()):
        logger.warning("Pre-computed files not found for %s", entry["event_id"])
        return None
    try:
        dl_samples = np.load(dl_file)
        df = pd.read_csv(gal_file)
    except Exception as exc:  # pragma: no cover - unexpected I/O errors
        logger.error("Failed loading pre-computed data for %s: %s", entry["event_id"], exc)
        raise
    return EventDataPackage(event_id=entry["event_id"], dl_samples=dl_samples, candidate_galaxies_df=df)


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
    loaded = _load_precomputed_data(event_cfg_entry)
    if loaded is not None:
        return loaded

    event_id = event_cfg_entry["event_id"]
    nside = event_cfg_entry.get("nside_skymap", NSIDE_SKYMAP)
    cdf = event_cfg_entry.get("cdf_threshold", CDF_THRESHOLD)
    catalog = event_cfg_entry.get("catalog_type", CATALOG_TYPE)
    z_fallback = event_cfg_entry.get("host_z_max_fallback", HOST_Z_MAX_FALLBACK)

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
    df = results["candidate_hosts_df"]
    return EventDataPackage(event_id=event_id, dl_samples=dl_samples, candidate_galaxies_df=df)


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
