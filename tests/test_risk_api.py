import pandas as pd

from app.api.routes_risk import compute_risk
from app.api.schemas import RiskRequest


def test_risk_endpoint_applies_slider_inputs(monkeypatch):
    def fake_get_calibration(tickers, horizon_days):
        class Calibration:
            def __init__(self):
                self.mu = pd.Series([0.01, 0.02], index=tickers, dtype=float)
                self.sigma = pd.DataFrame(
                    [[0.04, 0.01], [0.01, 0.09]],
                    index=tickers,
                    columns=tickers,
                    dtype=float,
                )
                self.cache_hit = True

        return Calibration()

    def fake_compute_risk_metrics(
        mu,
        sigma,
        weights,
        horizon_days,
        histogram_bins,
        tail_threshold,
    ):
        assert list(mu.index) == ["AAA", "BBB"]
        assert round(float(mu.iloc[0]), 6) == 0.0
        assert round(float(mu.iloc[1]), 6) == 0.01
        assert round(float(sigma.iloc[0, 0]), 6) > 0.04
        assert round(float(sigma.iloc[0, 1]), 6) > 0.01
        assert round(float(weights.sum()), 6) == 1.0
        assert horizon_days == 10
        assert histogram_bins == 30
        assert tail_threshold is None

        class Histogram:
            bin_edges = [0.0, 0.1, 0.2]
            counts = [3, 1]

        class Metrics:
            var = 0.12
            cvar = 0.15
            mean = 0.04
            vol = 0.08
            histogram = Histogram()

        return Metrics()

    monkeypatch.setattr("app.api.routes_risk.get_calibration", fake_get_calibration)
    monkeypatch.setattr(
        "app.api.routes_risk.compute_risk_metrics",
        fake_compute_risk_metrics,
    )

    response = compute_risk(
        RiskRequest(
            portfolio={"AAA": 0.6, "BBB": 0.4},
            horizon_days=10,
            return_model="normal",
            backend="classical",
            vol_multiplier=1.5,
            corr_spike=0.5,
            mean_shock=-0.01,
        )
    )

    assert response.var == 0.12
    assert response.backend.cache_hit is True


def test_risk_endpoint_uses_crash_slider_inputs(monkeypatch):
    def fake_get_calibration(tickers, horizon_days):
        class Calibration:
            def __init__(self):
                self.mu = pd.Series([0.01, 0.02], index=tickers, dtype=float)
                self.sigma = pd.DataFrame(
                    [[0.04, 0.0], [0.0, 0.09]],
                    index=tickers,
                    columns=tickers,
                    dtype=float,
                )
                self.cache_hit = False

        return Calibration()

    def fake_compute_risk_metrics(
        mu,
        sigma,
        weights,
        horizon_days,
        histogram_bins,
        tail_threshold,
    ):
        assert histogram_bins == 30

        class Histogram:
            bin_edges = [0.0, 0.1, 0.2]
            counts = [2, 2]

        class Metrics:
            var = 0.11
            cvar = 0.16
            mean = 0.05
            vol = 0.09
            histogram = Histogram()

        return Metrics()

    monkeypatch.setattr("app.api.routes_risk.get_calibration", fake_get_calibration)
    monkeypatch.setattr(
        "app.api.routes_risk.compute_risk_metrics",
        fake_compute_risk_metrics,
    )

    response = compute_risk(
        RiskRequest(
            portfolio={"AAA": 0.5, "BBB": 0.5},
            horizon_days=5,
            return_model="normal_crash_mixture",
            backend="classical",
            crash_pc=0.12,
            crash_mean_shift=-0.03,
            crash_vol_jump=3.0,
        )
    )

    assert response.backend.crash_params == {
        "pc": 0.12,
        "mean_shift": -0.03,
        "vol_jump": 3.0,
    }


def test_risk_endpoint_uses_qubits_to_set_quantum_bins(monkeypatch):
    def fake_get_calibration(tickers, horizon_days):
        class Calibration:
            def __init__(self):
                self.mu = pd.Series([0.01, 0.02], index=tickers, dtype=float)
                self.sigma = pd.DataFrame(
                    [[0.04, 0.01], [0.01, 0.09]],
                    index=tickers,
                    columns=tickers,
                    dtype=float,
                )
                self.cache_hit = True

        return Calibration()

    def fake_compute_risk_metrics(
        mu,
        sigma,
        weights,
        horizon_days,
        histogram_bins,
        tail_threshold,
    ):
        assert histogram_bins == 8

        class Histogram:
            bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            counts = [1, 1, 1, 1, 1, 1, 1, 1]

        class Metrics:
            var = 0.5
            cvar = 0.6
            mean = 0.25
            vol = 0.1
            histogram = Histogram()

        return Metrics()

    monkeypatch.setattr("app.api.routes_risk.get_calibration", fake_get_calibration)
    monkeypatch.setattr(
        "app.api.routes_risk.compute_risk_metrics",
        fake_compute_risk_metrics,
    )

    response = compute_risk(
        RiskRequest(
            portfolio={"AAA": 0.5, "BBB": 0.5},
            horizon_days=5,
            return_model="normal",
            backend="quantum",
            quantum_num_qubits=3,
        )
    )

    assert response.quantum is not None
    assert response.quantum.bin_qubits == 3
    assert response.quantum.n_bins == 8
    assert response.quantum.padded_bins == 8
