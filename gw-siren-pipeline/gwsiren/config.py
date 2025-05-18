from __future__ import annotations
from dataclasses import dataclass
import os
import pathlib

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
class Config:
    catalog: dict
    skymap: dict
    mcmc: dict
    cosmology: dict


def load_config(path: str | pathlib.Path | None = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, looks for config.yaml in root directory.
    """
    if path is not None:
        cfg_path = pathlib.Path(path)
    else:
        env_var = os.getenv("GWSIREN_CONFIG")
        if env_var:
            cfg_path = pathlib.Path(env_var)
        else:
            # Look for config.yaml in root directory (parent of gwsiren package)
            cfg_path = pathlib.Path(__file__).parent.parent / 'config.yaml'
        
    if not cfg_path.exists():
        raise RuntimeError(f'Config file not found: {cfg_path}')
    text = cfg_path.read_text()
    if _pyyaml is not None:
        raw = _pyyaml.safe_load(text)
    else:
        raw = _minimal_yaml_load(text)
    return Config(**raw)


CONFIG = load_config()
