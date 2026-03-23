---
theme: default
title: Portfolio RiskQ
info: Classical-to-Quantum Tail Risk Estimation
layout: cover
katex: true
---

<div class="cover">
  <div class="cover-logos">
    <img src="/Poster/wqc.png" alt="WQC" />
    <img src="/Poster/uwo.png" alt="UWO" />
  </div>
  <h1>Portfolio RiskQ</h1>
  <h2>Fast Tail Risk Estimation via Discretized IQAE</h2>
  <div class="cover-affil">
    <p class="cover-sub cover-names">Amir Orfali - Emily Berlinghoff</p>
    <p class="cover-sub cover-school">Western Quantum Club @ Western University</p>
  </div>
</div>

---
layout: two-cols
---
<div>
  <h1 class="center-title"> Motivation and Project Direction</h1>
</div>



<div class="card">
  <ul>
    <li>How may quantum algorithms intertwine with complex finance algorithms currently used?</li>
  </ul>
</div>



<div class="card accent">
  <ul>
    <li>Risk analysis (VaR, CVar)</li>
    <li>Use a cornerstone algorithm in quantatative finance</li>
    <li>Buld and run the algortihm classically</li>
    <li>Run the algorithm through a quantum computer simulation to show speedup</li>
  </ul>
</div>

::right::
<img src="/quantum-finance.webp" />

---
layout: two-cols
---

# Background & Definitions

<div class="card-lite">

- **VaR** = How much could we lose at most at X% of the time.  
- **CVaR** = tail risk of a risk distrubution.  
- Portfolio loss: $L = -w^T r$  
- Tail prob: $p = \mathbb{P}(L \ge \ell)$  
- Discretization maps losses into $2^n$ bins.

</div>

---
layout: two-cols
---

# Methodology (Math)

**Loss Model**  
$$ (\mu_s,\Sigma_s) \sim \mathcal{S},\quad r_{s,i} \sim \mathcal{N}(\mu_s,\Sigma_s),\quad L_{s,i} = -w^T r_{s,i} $$


**Tail Metrics**  
$$\mathrm{VaR}_\alpha = Q_\alpha(L),\quad \mathrm{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha]$$

**Discretization**  
$$p_i = n_i / N,\quad \sum_i p_i = 1,\quad \hat{p} = \sum_{i\in\text{tail}} p_i$$

**IQAE**  
$$p = \sin^2(\theta),\quad p_m = \sin^2((2m+1)\theta)$$

::right::

<div style="margin-top: 8rem;">

## Why it works

<div class="card">
<ul>
<li>Same ell and histogram ensure fair classical vs quantum comparison.</li>
<li>Discretization makes the tail probability quantum-ready.</li>
<li>IQAE uses Grover amplification to infer p efficiently.</li>
</ul>
</div>

</div>

---
class: white-bg
---

<h1 class="center-title">Pipeline (Classical → Quantum)</h1>

<div class="pipeline-graphic">
  <div class="pl node classic">
    <div class="bubble">D</div>
    <div class="label">Market Data</div>
  </div>
  <div class="pl arrow">→</div>
  <div class="pl node classic">
    <div class="bubble">M</div>
    <div class="label">Calibration (μ, σ)</div>
  </div>
  <div class="pl arrow">→</div>
  <div class="pl node classic">
    <div class="bubble">MC</div>
    <div class="label">Nested MC</div>
  </div>
  <div class="pl arrow">→</div>
  <div class="pl node classic">
    <div class="bubble">UI</div>
    <div class="label">Histogram 2<sup>n</sup></div>
  </div>
  <div class="pl arrow">→</div>
  <div class="pl node quantum">
    <div class="bubble q">Q</div>
    <div class="label">IQAE</div>
  </div>

  <div class="pl arrow-down sub-arrow-var">↓</div>
  <div class="pl node classic sub sub-var">
    <div class="label">VaR / CVaR</div>
  </div>
  <div class="pl arrow-down sub-arrow-tail">↓</div>
  <div class="pl node classic sub sub-tail">
    <div class="label">Tail Mask</div>
  </div>
</div>

<div class="pipeline-text">

- **Calibration:** estimate $\mu,\Sigma$ (and stress modifiers if enabled).  
- **Nested MC:** outer loop samples regimes; inner loop draws $r\sim\mathcal{N}(\mu_s,\Sigma_s)$ and losses $L=-w^\top r$.  
- **Discretization:** map losses into $2^n$ bins, normalize probabilities, build tail mask at $\ell$.  
- **State prep:** encode histogram amplitudes; tail mask defines “good” states.  
- **IQAE:** Grover iterations estimate tail probability $\hat{p}$ with confidence interval and resource report.  
- **Alignment:** same $\ell$, same histogram, same scenarios for classical vs quantum.

</div>

---
class: white-bg
---
<h1 class="center-title">Dashboard Validation</h1>

<div class="dash-grid">
  <iframe src="http://localhost:8501" style="width:300%; height:140vh; border:0; border-radius:12px; transform: scale(0.34); transform-origin: top left;"></iframe>
  <iframe src="http://localhost:8501" style="width:300%; height:140vh; border:0; border-radius:12px; transform: scale(0.34); transform-origin: top left;"></iframe>
</div>
<div class="caption">Live dashboard view (requires Streamlit running locally).</div>

---
class: white-bg
---
<div class="exp-page">
  <h1 class="center-title">Experiments</h1>
  <div class="exp-grid">
    <div class="exp-stack">
      <div class="exp-card flip-wrap">
        <input id="flip-1" class="flip-toggle" type="checkbox" checked />
        <label for="flip-1" class="flip-card">
          <div class="flip-face flip-front">
            <object data="/experiments/animated/classical_error_vs_paths.svg" type="image/svg+xml" class="exp-svg"></object>
            <div class="caption">VaR error falls roughly as 1/√N; 10k+ paths stabilize tails.</div>
          </div>
          <div class="flip-face flip-back">
          <div class="flip-title">How fast does MC error drop as we add paths?</div>
          </div>
        </label>
      </div>
      <div class="exp-card flip-wrap">
        <input id="flip-2" class="flip-toggle" type="checkbox" checked />
        <label for="flip-2" class="flip-card">
          <div class="flip-face flip-front">
            <object data="/experiments/animated/error_vs_discretization_bits.svg" type="image/svg+xml" class="exp-svg"></object>
            <div class="caption">Gains taper beyond 6–7 bits (binning bias).</div>
          </div>
          <div class="flip-face flip-back">
          <div class="flip-title">How many bins do we need before gains taper?</div>
          </div>
        </label>
      </div>
    </div>
    <div class="exp-stack">
      <div class="exp-card flip-wrap">
        <input id="flip-3" class="flip-toggle" type="checkbox" checked />
        <label for="flip-3" class="flip-card">
          <div class="flip-face flip-front">
            <object data="/experiments/animated/quantum_error_vs_oracle_calls.svg" type="image/svg+xml" class="exp-svg"></object>
            <div class="caption">Sub‑1% tail error in a few Grover rounds.</div>
          </div>
          <div class="flip-face flip-back">
          <div class="flip-title">How many oracle calls deliver most of the accuracy?</div>
          </div>
        </label>
      </div>
      <div class="exp-card flip-wrap">
        <input id="flip-4" class="flip-toggle" type="checkbox" checked />
        <label for="flip-4" class="flip-card">
          <div class="flip-face flip-front">
            <object data="/experiments/animated/error_vs_noise_level.svg" type="image/svg+xml" class="exp-svg"></object>
            <div class="caption">Noise (σ >= 0.05) dominates error.</div>
          </div>
          <div class="flip-face flip-back">
          <div class="flip-title">At what noise level does estimation break down?</div>
          </div>
        </label>
      </div>
    </div>
  </div>
</div>

---
---

<h1 class="center-title full-width-title">Feasibility Boundary</h1>

<div class="two-col">
  <div>
    <img class="full" src="/experiments/plots/feasibility_boundary.png" />
  </div>
  <div>
    <p><span class="green">Feasible</span> settings stay within near-term resource limits, <span class="yellow">borderline</span> settings become costly quickly, and <span class="red">not viable</span> regions exceed NISQ-era limits. <span class="caption">Colour bands use a composite resource score. The score averages qubit count, circuit depth, and oracle calls to reflect overall hardware load.</span></p>
  </div>
</div>

---

<h1 class="center-title">Classical MC vs Quantum (IQAE)</h1>

<div class="card">

| | **Classical MC** | **Quantum (IQAE)** |
|---|---|---|
| **Accuracy scaling** | Improves as you add paths | Improves with Grover iterations |
| **Cost driver** | Number of Monte Carlo paths | Oracle calls / circuit depth |
| **Tail probability** | Estimated directly from samples | Estimated via amplitude estimation |
| **Interpretability** | High, sample‑based | High, same histogram + tail mask |
| **Feasibility limit** | Compute time | Qubits, depth, and oracle budget |

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

# Who is pursuing this currently?

<div class="card">
<p><strong>J.P Morgan Chase, Goldman Sachs, HSBC, Barclays, BNP Paribas </strong></p>
<p></p>
</div>

<div class="card-lite">
<p><strong>How are they using quantum?</strong></p>
<ul>
<li>Building quantum algorithms internally for pricing and risk</li>
<li>JPMorgan built and ran actual quantum algorithms on hardware (Barron)</li>
<li>IBM + JPMorgan are testing portfolio optimization + derivatives pricing algorithms together</li>
</ul>
</div>

<img src="/qaoa-media-preview.png" style="display:block; max-width:34%; max-height:34vh; object-fit:contain; margin:0.75rem auto 0;" />


---

<h1 class="center-title">References</h1>

- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*.
- Jorion, P. (2007). *Value at Risk*.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
- Brassard, G. et al. (2002). Quantum Amplitude Estimation and its Applications. *Contemporary Mathematics*.
- Grinko, D. et al. (2021). Iterative Quantum Amplitude Estimation. *npj Quantum Information*.
- Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. *Proceedings of STOC*.

---
class: center-slide
---

<div class="center-slide">
  <h1 class="center-title">Thank You</h1>
  <div class="contact">
    <p>Western Quantum Club - Western University</p>
    <p>aorfali2@uwo.ca - eberling@uwo.ca</p>
    <p>github.com/amirorfali/Dynamic-Risk-Dashboard</p>
  </div>
</div>
