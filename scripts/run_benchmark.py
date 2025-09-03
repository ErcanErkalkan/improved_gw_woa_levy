#!/usr/bin/env python
import argparse, yaml, numpy as np
from pathlib import Path
from egwwoa.core.fitness import toy_fitness_factory
from egwwoa.core.rng import make_rng
from egwwoa.algorithms.base import AlgoConfig
from egwwoa.algorithms.egwwoa import EGW_WOA
from egwwoa.algorithms.gwo import GWO
from egwwoa.algorithms.woa import WOA
from egwwoa.algorithms.base_gwwoa import GW_WOA_Base

def build(name, acfg, fitness, lb, ub, rng, cfg):
    if name == "EGW-WOA":
        p = cfg["egwwoa"]
        return EGW_WOA(acfg, fitness, lb, ub, rng,
                       p_levy=p["p_levy"], beta_L=p["beta_L"], b_spiral=p["b_spiral"],
                       I_stall=p["I_stall"], phi=p["phi"],
                       omega_max=p["omega"]["max"], omega_min=p["omega"]["min"])
    if name == "GWO": return GWO(acfg, fitness, lb, ub, rng)
    if name == "WOA": return WOA(acfg, fitness, lb, ub, rng, b_spiral=cfg["egwwoa"]["b_spiral"])
    if name in ("GW-WOA-base","GW--WOA (base)"): return GW_WOA_Base(acfg, fitness, lb, ub, rng, b_spiral=cfg["egwwoa"]["b_spiral"])
    return GWO(acfg, fitness, lb, ub, rng)

def main(cfg_path:str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    T_hor = int(cfg.get("T_hor", 24))
    N = int(cfg.get("population_size", 30))
    T_iter = int(cfg.get("T_iter", 50))
    seed = int(cfg.get("seed", 123))
    rng = make_rng(seed)
    fitness, D = toy_fitness_factory(T_hor=T_hor)
    lb, ub = -5*np.ones(D), 5*np.ones(D)
    acfg = AlgoConfig(N=N, T_iter=T_iter)
    res = {}
    for name in cfg["algorithms"]:
        algo = build(name, acfg, fitness, lb, ub, rng, cfg)
        out = algo.run()
        res[name] = out["f_best"]
    print("Summary (toy):")
    for k, v in res.items():
        print(f"{k:12s} : {v:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    main(ap.parse_args().config)
