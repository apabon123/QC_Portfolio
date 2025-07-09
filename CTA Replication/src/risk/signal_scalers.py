import numpy as np
from typing import Dict, Optional

class ZScoreSignalScaler:
    """Convert qualified momentum signals into forecasts using Z-Score normalisation.

    Provides detailed diagnostic logging when a QuantConnect *algorithm* instance
    is supplied.  The log mirrors the step-by-step description requested:

    1. Risk-adjusted signal (already supplied by strategy)
    2. Mean / standard deviation
    3. Z-score, symmetric caps (±cap_threshold)
    4. Final forecast (capped_z / N)
    """

    def __init__(self, cap_threshold: float = 1.60, algorithm: Optional[object] = None, name: str = ""):
        self.cap_threshold = cap_threshold
        self.algorithm = algorithm
        self.name = name or "ZScoreScaler"

    def process(self, qualified_signals: Dict) -> Dict:
        """Return a dict of forecasts keyed by asset symbol."""
        if not qualified_signals:
            return {}

        assets = list(qualified_signals.keys())
        signals = np.array(list(qualified_signals.values()), dtype=float)

        # When dispersion is zero we cannot form a Z-score – return zeros.
        std = float(np.std(signals))
        if std == 0.0:
            if self.algorithm:
                self.algorithm.Debug(f"{self.name}: Dispersion zero – all forecasts set to 0")
            return {asset: 0.0 for asset in assets}

        mean = float(np.mean(signals))
        z_scores = (signals - mean) / std

        capped = np.clip(z_scores, -self.cap_threshold, self.cap_threshold)
        n_assets = len(assets)
        forecasts = capped / n_assets

        if self.algorithm:
            for asset, raw, z, cap_val, fore in zip(assets, signals, z_scores, capped, forecasts):
                self.algorithm.Debug(
                    f"{self.name}: {asset} | raw={raw:.3f}, z={z:.3f}, capped={cap_val:.3f}, forecast={fore:.3f}")
        return dict(zip(assets, forecasts))


class MADSignalScaler:
    """Convert qualified momentum signals into forecasts using Carver's MAD scaling.

    Implements the four-step process with optional detailed diagnostic logs:

    1. Risk-adjusted signal
    2. Scalar = 10 / mean(abs(signal))
    3. Scale & cap to ±20
    4. Forecast = capped / 10 / N
    """

    def __init__(self,
                 target_mad: float = 10.0,
                 cap_upper: float = 20.0,
                 cap_lower: float = -20.0,
                 algorithm: Optional[object] = None,
                 name: str = ""):
        self.target_mad = target_mad
        self.cap_upper = cap_upper
        self.cap_lower = cap_lower
        self.algorithm = algorithm
        self.name = name or "MADScaler"

    def process(self, qualified_signals: Dict) -> Dict:
        """Return a dict of forecasts keyed by asset symbol."""
        if not qualified_signals:
            return {}

        assets = list(qualified_signals.keys())
        signals = np.array(list(qualified_signals.values()), dtype=float)

        # MAD is the mean absolute deviation from zero.
        mad = float(np.mean(np.abs(signals)))
        if mad == 0.0:
            if self.algorithm:
                self.algorithm.Debug(f"{self.name}: Zero MAD – all forecasts set to 0")
            return {asset: 0.0 for asset in assets}

        # Scale the signals such that the resulting MAD equals ``target_mad``.
        scalar = self.target_mad / mad
        scaled = signals * scalar

        # Apply fixed caps – this introduces adaptive asymmetry around the mean.
        capped = np.clip(scaled, self.cap_lower, self.cap_upper)

        # Convert back to forecasts in roughly the same range as Z-Score (≈ ±2)
        forecasts = capped / self.target_mad / len(assets)

        if self.algorithm:
            for asset, raw, sc, cap_val, fore in zip(assets, signals, scaled, capped, forecasts):
                self.algorithm.Debug(
                    f"{self.name}: {asset} | raw={raw:.3f}, scalar={scalar:.3f}, scaled={sc:.3f}, "
                    f"capped={cap_val:.3f}, forecast={fore:.3f}")
        return dict(zip(assets, forecasts)) 