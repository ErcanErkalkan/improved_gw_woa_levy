# EGW–WOA (Enhanced Gray Wolf–Whale Optimization) – Project

Implements the framework from the manuscript: **Integrating GWO, WOA, Lévy Flights, and Chaotic Re-initialization**.

**Notation is consistent**: `T_iter` (iteration budget), `T_hor` (horizon hours), `beta_L`, `p_levy`, `I_stall`, `phi`, `omega(t)`.

## Quickstart
```bash
conda env create -f environment.yml
conda activate egw-woa
python scripts/run_benchmark.py --config configs/main.yaml
```
