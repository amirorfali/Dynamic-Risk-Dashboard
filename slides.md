---
theme: default
title: Portfolio RiskQ
info: Classical-to-Quantum Tail Risk Estimation
layout: cover
---

<div class="cover">
  <div class="cover-logos">
    <img src="/Poster/wqc.png" alt="WQC" />
    <img src="/Poster/uwo.png" alt="UWO" />
  </div>
  <h1>Portfolio RiskQ</h1>
  <h2>Fast Tail Risk Estimation via Discretized IQAE</h2>
  <p class="cover-sub">Emily Berlinghoff - Amir Orfali</p>
  <p class="cover-sub">Western University - Western Quantum Club</p>
</div>

---
layout: two-cols
---

# Problem & Goal

<div class="card">
<ul>
<li>Risk teams need VaR/CVaR accuracy under tight compute budgets.</li>
<li>Classical MC is robust but expensive for tail probabilities.</li>
<li>Goal: preserve interpretability while enabling faster tail estimation.</li>
</ul>
</div>

::right::

# Key Contribution

<div class="card accent">
<ul>
<li>Classical -> quantum VaR/CVaR workflow.</li>
<li>Discretized histogram + tail mask bridge.</li>
<li>IQAE vs nested Monte Carlo comparison.</li>
<li>Feasibility boundary for NISQ constraints.</li>
<li>Production-style dashboard validation.</li>
</ul>
</div>

---
layout: two-cols
---

# Background & Definitions

<div class="card-lite">

- **VaR** = loss quantile at confidence alpha.  
- **CVaR** = expected loss beyond VaR.  
- Portfolio loss: $L = -w^T r$  
- Tail prob: $p = \mathbb{P}(L \ge \ell)$  
- Discretization maps losses into $2^n$ bins.

</div>
::right::

# Pipeline (Classical -> Quantum)

<div class="pipeline">
  <div class="node classic">Market Data</div>
  <div class="arrow">-></div>
  <div class="node classic">Calibration (mu, Sigma)</div>
  <div class="arrow">-></div>
  <div class="node classic">Nested MC</div>
  <div class="arrow">-></div>
  <div class="node classic">Histogram 2^n</div>
  <div class="arrow">-></div>
  <div class="node quantum">IQAE</div>
</div>

<div class="caption">
Classical stages are in green; IQAE is in purple.
</div>

---
layout: two-cols
---

# Methodology (Math)

**Loss model**  
$$r \sim \mathcal{N}(\mu, \Sigma),\quad L = -w^T r$$

**Tail metrics**  
$$\mathrm{VaR}_\alpha = Q_\alpha(L),\quad \mathrm{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha]$$

**Discretization**  
$$p_i = n_i / N,\quad \sum_i p_i = 1,\quad \hat{p} = \sum_{i\in\text{tail}} p_i$$

**IQAE**  
$$p = \sin^2(\theta),\quad p_m = \sin^2((2m+1)\theta)$$

::right::

# Why it works

<div class="card">
<ul>
<li>Same ell and histogram ensure fair classical vs quantum comparison.</li>
<li>Discretization makes the tail probability quantum-ready.</li>
<li>IQAE uses Grover amplification to infer p efficiently.</li>
</ul>
</div>

---
# Experiments

<div class="grid">
  <div class="plot-card">
    <img src="/experiments/plots/classical_error_vs_paths.png" />
    <div class="badge">Takeaway</div>
    <div class="caption">VaR error falls with 1/sqrt(N); 10k+ paths stabilizes tails.</div>
  </div>
  <div class="plot-card">
    <img src="/experiments/plots/quantum_error_vs_oracle_calls.png" />
    <div class="badge">Takeaway</div>
    <div class="caption">Sub-1% tail error in a few Grover rounds.</div>
  </div>
  <div class="plot-card">
    <img src="/experiments/plots/error_vs_discretization_bits.png" />
    <div class="badge">Takeaway</div>
    <div class="caption">Gains taper beyond 6-7 bits (binning bias).</div>
  </div>
  <div class="plot-card">
    <img src="/experiments/plots/error_vs_noise_level.png" />
    <div class="badge">Takeaway</div>
    <div class="caption">Noise (sigma >= 0.05) dominates error.</div>
  </div>
</div>

---
layout: two-cols
---

# Feasibility Boundary

<img class="full" src="/experiments/plots/feasibility_boundary.png" />

::right::

<div class="card">
<p><span class="green">Feasible</span> settings are within near-term resource limits.</p>
<p><span class="yellow">Borderline</span> settings become costly quickly.</p>
<p><span class="red">Not viable</span> regions exceed NISQ-era limits.</p>
<p class="caption">Bands use a composite resource score (qubits + depth + oracle calls).</p>
</div>

---
# Dashboard Validation

<div class="grid-2">
  <div class="plot-card">
    <div class="plot-title">Classical Backend</div>
    <img src="/Poster/c_dash.png" />
    <div class="caption">Interpretability: tail-highlighted histogram + VaR/CVaR.</div>
  </div>
  <div class="plot-card">
    <div class="plot-title">Quantum (IQAE) Backend</div>
    <img src="/Poster/q_dash.png" />
    <div class="caption">Usability: IQAE CI + absolute/relative error vs classical.</div>
  </div>
</div>

---
layout: two-cols
---

# Results & Implications

<div class="card">
<ul>
<li>IQAE estimates matched classical tail probability on toy data.</li>
<li>Discretization preserved tail structure and enabled fair comparisons.</li>
<li>Resource reports exposed the feasible operating region.</li>
</ul>
</div>

::right::

# Conclusion

<div class="card accent">
<p>We validated a classical-to-quantum risk pipeline with consistent inputs, interpretable outputs, and a clear audit trail from data to tail estimate.</p>
</div>

---
layout: center
---

# References

- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*.
- Jorion, P. (2007). *Value at Risk*.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
- Brassard, G. et al. (2002). Quantum Amplitude Estimation and its Applications. *Contemporary Mathematics*.
- Grinko, D. et al. (2021). Iterative Quantum Amplitude Estimation. *npj Quantum Information*.
- Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. *Proceedings of STOC*.

---
layout: center
---

# Thank You

<div class="contact">
<p>Western Quantum Club - Western University</p>
<p>aorfali2@uwo.ca - eberling@uwo.ca</p>
<p>github.com/amirorfali/Dynamic-Risk-Dashboard</p>
</div>
