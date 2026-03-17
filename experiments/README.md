# Experiments

These scripts generate plots for the final report/demo and poster visuals.

## Quick run

```bash
./venv/bin/python experiments/run_experiments.py
```

Plots are saved to `experiments/plots/`:
- `classical_error_vs_paths.png`
- `quantum_error_vs_oracle_calls.png`
- `error_vs_discretization_bits.png`
- `error_vs_noise_level.png`
- `feasibility_boundary.png`

## Notes
- All experiments use a standard normal loss model with tail threshold at VaR 99%.
- The feasibility chart uses heuristic thresholds for green/yellow/red bands.
- IQAE experiments use the simulator in `app/core/quantum_ae.py`.
