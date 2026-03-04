"""NS-3 configuration helper for topology + PHY settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Ns3Config:
    wifi_standard: str = "80211ac"
    channel_width_mhz: int = 20
    tx_power_start_dbm: float = 20.0
    tx_power_end_dbm: float = 30.0


class Ns3Bridge:
    def __init__(self, config: Ns3Config | None = None):
        self.config = config or Ns3Config()

    def export_topology(self, positions: np.ndarray, link_matrix: np.ndarray) -> Dict:
        return {
            "nodes": [{"id": i, "position": positions[i].tolist()} for i in range(len(positions))],
            "links": [
                {"src": int(i), "dst": int(j)}
                for i in range(link_matrix.shape[0])
                for j in range(link_matrix.shape[1])
                if link_matrix[i, j] > 0
            ],
            "phy": self.config.__dict__,
        }

    def export_power_schedule(self, tx_powers_w: np.ndarray) -> Dict:
        tx_dbm = (10.0 * np.log10(np.maximum(tx_powers_w, 1e-9)) + 30.0).tolist()
        return {"tx_power_dbm_per_node": tx_dbm}
