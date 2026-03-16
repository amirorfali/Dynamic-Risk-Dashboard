import json
from datetime import datetime
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

st.set_page_config(page_title="Risk Dashboard", layout="wide")

st.markdown(
    """
<style>
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        color: #e2e8f0;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
    }
    .panel {
        background: #0b1120;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 18px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Dynamic Risk Dashboard")
st.write("MVP UI – connect to the FastAPI backend and inspect results.")

with st.expander("API Settings", expanded=True):
    api_url = st.text_input(
        "Risk endpoint URL",
        value="http://127.0.0.1:8000/api/risk",
    )

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Request")
    portfolio_input = st.text_input(
        "Portfolio (ticker:weight, comma-separated)",
        value="AAPL:0.5, MSFT:0.3, GOOGL:0.2",
    )
    horizon_days = st.number_input("Horizon days", min_value=1, value=10, step=1)
    return_model = st.selectbox(
        "Return model",
        options=["normal", "normal_crash_mixture"],
        index=0,
    )
    tail_threshold = st.text_input("Tail threshold (optional)", value="")

    st.subheader("Stress sliders")
    vol_multiplier = st.slider("Volatility multiplier", 0.5, 3.0, 1.0, 0.1)
    corr_spike = st.slider("Correlation spike", 0.0, 1.0, 0.0, 0.05)
    mean_shock = st.slider("Mean shock", -0.05, 0.05, 0.0, 0.005)
    crash_pc = st.slider("Crash probability", 0.0, 0.2, 0.05, 0.01)
    crash_mean_shift = st.slider("Crash mean shift", -0.1, 0.0, -0.02, 0.005)
    crash_vol_jump = st.slider("Crash vol jump", 1.0, 5.0, 2.0, 0.1)

    st.subheader("Backend toggle")
    backend_mode = st.radio(
        "Computation mode",
        ["Classical", "Quantum (IQAE)"],
        horizontal=True,
    )


def _parse_portfolio(raw: str) -> dict[str, float]:
    if not raw.strip():
        return {}
    items = [item.strip() for item in raw.split(",") if item.strip()]
    portfolio = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid entry '{item}', expected TICKER:WEIGHT")
        ticker, weight = item.split(":", 1)
        portfolio[ticker.strip().upper()] = float(weight.strip())
    return portfolio


payload = {
    "portfolio": {},
    "horizon_days": int(horizon_days),
    "return_model": return_model,
    "backend": "quantum" if backend_mode == "Quantum (IQAE)" else "classical",
}
if tail_threshold.strip():
    payload["tail_threshold"] = float(tail_threshold)

try:
    payload["portfolio"] = _parse_portfolio(portfolio_input)
    payload_error = None
except ValueError as exc:
    payload_error = str(exc)

with left:
    st.code(json.dumps(payload, indent=2), language="json")

    if payload_error:
        st.error(payload_error)

    run = st.button("Run Risk", use_container_width=True)

with right:
    st.subheader("Results")
    if run:
        if payload_error:
            st.stop()
        try:
            req = Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            st.error(f"HTTP error: {exc.code} {exc.reason}")
            st.stop()
        except URLError as exc:
            st.error(f"Connection error: {exc.reason}")
            st.stop()
        except json.JSONDecodeError:
            st.error("Failed to parse response JSON.")
            st.stop()

        metrics = [
            ("VaR", data.get("var")),
            ("CVaR", data.get("cvar")),
            ("Mean", data.get("mean")),
            ("Vol", data.get("vol")),
        ]
        cols = st.columns(4)
        for col, (label, value) in zip(cols, metrics):
            display = "—" if value is None else f"{value:.4f}"
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{display}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        backend = data.get("backend", {})
        cache_hit = backend.get("cache_hit")
        runtime_ms = backend.get("runtime_ms")
        data_source = backend.get("data_source")
        window_days = backend.get("window_days")
        crash_params = backend.get("crash_params")

        cache_label = "—" if cache_hit is None else ("Hit" if cache_hit else "Miss")
        runtime_label = "—" if runtime_ms is None else f"{runtime_ms:.1f}"
        window_label = "—" if window_days is None else f"{window_days}d"

        info_cols = st.columns(3)
        info_cols[0].markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Cache</div>"
            f"<div class='metric-value'>{cache_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        info_cols[1].markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Runtime (ms)</div>"
            f"<div class='metric-value'>{runtime_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        info_cols[2].markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Data Window</div>"
            f"<div class='metric-value'>{window_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption(
            f"Source: {data_source or 'unknown'} · "
            f"Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        histogram = data.get("histogram", {})
        edges = histogram.get("bin_edges", [])
        counts = histogram.get("counts", [])
        if len(edges) >= 2 and len(counts) == len(edges) - 1:
            centers = [
                (edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)
            ]
            tail_cut = data.get("tail_threshold")
            if not isinstance(tail_cut, (int, float)):
                tail_cut = data.get("var")
            tail_mask = (
                [c >= tail_cut for c in centers] if isinstance(tail_cut, (int, float)) else [False] * len(centers)
            )
            chart = pd.DataFrame({"loss": centers, "count": counts})
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=chart["loss"],
                        y=chart["count"],
                        marker_color=[
                            "#fb7185" if is_tail else "#38bdf8" for is_tail in tail_mask
                        ],
                        opacity=0.9,
                    )
                ]
            )
            var_val = data.get("var")
            cvar_val = data.get("cvar")
            if isinstance(var_val, (int, float)):
                fig.add_vline(
                    x=var_val,
                    line_width=2,
                    line_dash="dash",
                    line_color="#f97316",
                    annotation_text="VaR",
                    annotation_position="top right",
                )
            if isinstance(cvar_val, (int, float)):
                fig.add_vline(
                    x=cvar_val,
                    line_width=2,
                    line_dash="dot",
                    line_color="#e11d48",
                    annotation_text="CVaR",
                    annotation_position="top right",
                )
            fig.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=40, b=30),
                title="Loss Histogram",
                xaxis_title="Loss",
                yaxis_title="Count",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feasibility panel")
        weight_sum = sum(payload.get("portfolio", {}).values())
        if payload.get("portfolio"):
            st.write(f"Portfolio weight sum: {weight_sum:.3f}")
            if abs(weight_sum - 1.0) > 0.01:
                st.warning("Portfolio weights do not sum to 1.0.")
        if not tail_threshold.strip():
            st.info("Tail threshold not provided; backend defaults to VaR at 99%.")

        if any([vol_multiplier != 1.0, corr_spike != 0.0, mean_shock != 0.0]) or crash_pc != 0.05:
            st.warning("Stress sliders are not wired to the backend yet.")

        quantum = data.get("quantum")
        if backend_mode == "Quantum (IQAE)" and quantum:
            tail_prob = quantum.get("tail_prob")
            estimate = quantum.get("estimate")
            ci_low = quantum.get("ci_low")
            ci_high = quantum.get("ci_high")
            diff_abs = quantum.get("diff_abs")
            diff_rel = quantum.get("diff_rel")
            padded_bins = quantum.get("padded_bins")
            n_bins = quantum.get("n_bins")

            if padded_bins and n_bins and padded_bins != n_bins:
                st.warning(f"Histogram bins padded from {n_bins} to {padded_bins} for IQAE.")

            st.subheader("Classical vs Quantum")
            comp_cols = st.columns(3)
            classical_display = "—" if tail_prob is None else f"{tail_prob:.4f}"
            quantum_display = "—"
            if estimate is not None and ci_low is not None and ci_high is not None:
                quantum_display = f"{estimate:.4f} ({ci_low:.4f}, {ci_high:.4f})"
            diff_abs_display = "—" if diff_abs is None else f"{diff_abs:.4f}"
            diff_rel_display = "—" if diff_rel is None else f"{diff_rel:.2%}"

            comp_cols[0].markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Classical Tail Prob</div>"
                f"<div class='metric-value'>{classical_display}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            comp_cols[1].markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>IQAE Estimate (CI)</div>"
                f"<div class='metric-value'>{quantum_display}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            comp_cols[2].markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Abs / Rel Diff</div>"
                f"<div class='metric-value'>{diff_abs_display} | {diff_rel_display}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if crash_params:
            st.subheader("Crash Mixture Parameters")
            st.json(crash_params)

        with st.expander("Raw JSON"):
            st.json(data)
