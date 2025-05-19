from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventDataPackage:
    """Container for prepared event data used in multi-event analyses."""

    event_id: str
    dl_samples: np.ndarray
    candidate_galaxies_df: pd.DataFrame


__all__ = ["EventDataPackage"]
