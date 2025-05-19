from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pathlib
import os

# Try to import PyYAML, fall back to a very small parser if unavailable
try:
    import yaml as _pyyaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML missing
    _pyyaml = None


def _minimal_yaml_load(text: str) -> dict:
    """Very small YAML loader supporting two-level mappings."""
    data: dict[str, dict[str, object]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(' '):
            key = line.rstrip(':').strip()
            data[key] = {}
            current = data[key]
        else:
            if current is None:
                raise ValueError('Invalid YAML structure')
            sub_key, value = line.strip().split(':', 1)
            value = value.strip()
            if value == '':
                val_obj: object = None
            else:
                try:
                    val_obj = int(value)
                except ValueError:
                    try:
                        val_obj = float(value)
                    except ValueError:
                        val_obj = value
            current[sub_key] = val_obj
    return data


@dataclass(frozen=True)
class MEInitialPosParam:
    """Initial position configuration for a parameter."""

    mean: float
    std: float


@dataclass(frozen=True)
class MEMCMCConfig:
    """MCMC configuration specific to multi-event analysis."""

    n_walkers: Optional[int] = None
    n_steps: Optional[int] = None
    burnin: Optional[int] = None
    thin_by: Optional[int] = None
    initial_pos_config: Optional[Dict[str, MEInitialPosParam]] = None


@dataclass(frozen=True)
class MECosmologyConfig:
    """Cosmology configuration overrides for multi-event analysis."""

    sigma_v_pec: Optional[float] = None
    c_light: Optional[float] = None
    omega_m_val: Optional[float] = None


@dataclass(frozen=True)
class MEPriorBoundaries:
    """Prior bounds for a parameter."""

    min: float
    max: float


@dataclass(frozen=True)
class MEEventToCombine:
    """Single event entry within the multi-event configuration."""

    event_id: str
    single_event_processing_params: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class MERunSettings:
    """General run settings for multi-event analysis."""

    run_label: Optional[str] = None
    base_output_directory: Optional[str] = None
    candidate_galaxy_cache_dir: Optional[str] = "cache/candidate_galaxies/"
    gw_posteriors_cache_dir: Optional[str] = "cache/gw_posteriors/"


@dataclass(frozen=True)
class MultiEventAnalysisSettings:
    """Top-level container for multi-event analysis settings."""

    run_settings: Optional[MERunSettings] = None
    events_to_combine: List[MEEventToCombine] = field(default_factory=list)
    priors: Optional[Dict[str, MEPriorBoundaries]] = None
    mcmc: Optional[MEMCMCConfig] = None
    cosmology: Optional[MECosmologyConfig] = None


def _parse_multi_event_config(raw: Dict) -> MultiEventAnalysisSettings:
    """Convert a raw dictionary into ``MultiEventAnalysisSettings``."""

    run_raw = raw.get("run_settings")
    run_cfg = MERunSettings(**run_raw) if isinstance(run_raw, dict) else None

    events_cfg: List[MEEventToCombine] = []
    events_raw = raw.get("events_to_combine", [])
    if not isinstance(events_raw, list):
        raise ValueError("'events_to_combine' must be a list")
    for entry in events_raw:
        if not isinstance(entry, dict) or "event_id" not in entry:
            raise ValueError("Each event entry must contain 'event_id'")
        events_cfg.append(
            MEEventToCombine(
                event_id=entry["event_id"],
                single_event_processing_params=entry.get(
                    "single_event_processing_params"
                ),
            )
        )

    priors_raw = raw.get("priors")
    priors_cfg = None
    if isinstance(priors_raw, dict):
        priors_cfg = {
            key: MEPriorBoundaries(**val) for key, val in priors_raw.items()
        }

    mcmc_raw = raw.get("mcmc")
    mcmc_cfg = None
    if isinstance(mcmc_raw, dict):
        init_raw = mcmc_raw.get("initial_pos_config")
        init_cfg = None
        if isinstance(init_raw, dict):
            init_cfg = {
                k: MEInitialPosParam(**v) for k, v in init_raw.items()
            }
        mcmc_cfg = MEMCMCConfig(
            n_walkers=mcmc_raw.get("n_walkers"),
            n_steps=mcmc_raw.get("n_steps"),
            burnin=mcmc_raw.get("burnin"),
            thin_by=mcmc_raw.get("thin_by"),
            initial_pos_config=init_cfg,
        )

    cosmo_raw = raw.get("cosmology")
    cosmo_cfg = None
    if isinstance(cosmo_raw, dict):
        cosmo_cfg = MECosmologyConfig(
            sigma_v_pec=cosmo_raw.get("sigma_v_pec"),
            c_light=cosmo_raw.get("c_light"),
            omega_m_val=cosmo_raw.get("omega_m_val"),
        )

    return MultiEventAnalysisSettings(
        run_settings=run_cfg,
        events_to_combine=events_cfg,
        priors=priors_cfg,
        mcmc=mcmc_cfg,
        cosmology=cosmo_cfg,
    )

@dataclass(frozen=True)
class Config:
    """Typed container for configuration sections."""

    catalog: dict
    skymap: dict
    mcmc: dict
    cosmology: dict
    fetcher: dict
    multi_event_analysis: Optional[MultiEventAnalysisSettings] = None


def load_config(path: str | pathlib.Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Optional explicit path to the config file.
        If ``None`` and the ``GWSIREN_CONFIG`` environment variable is set, use
        that path. Otherwise look for ``config.yaml`` next to the package.
    """
    if path is None:
        env_cfg = os.environ.get('GWSIREN_CONFIG')
        if env_cfg:
            cfg_path = pathlib.Path(env_cfg)
        else:
            cfg_path = pathlib.Path(__file__).parent.parent / 'config.yaml'
    else:
        cfg_path = pathlib.Path(path)
        
    if not cfg_path.exists():
        raise RuntimeError(f'Config file not found: {cfg_path}')
    text = cfg_path.read_text()
    if _pyyaml is not None:
        raw = _pyyaml.safe_load(text)
    else:
        raw = _minimal_yaml_load(text)
    me_raw = raw.get("multi_event_analysis")
    if me_raw is not None:
        raw.pop("multi_event_analysis")
        me_cfg = _parse_multi_event_config(me_raw)
    else:
        me_cfg = None
    return Config(**raw, multi_event_analysis=me_cfg)


CONFIG = load_config()
