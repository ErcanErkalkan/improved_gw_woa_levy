#!/usr/bin/env python
import argparse, yaml, numpy as np
from pathlib import Path
from egwwoa.core.fitness import toy_fitness_factory
from egwwoa.core.rng import make_rng
from egwwoa.algorithms.base import AlgoConfig
from egwwoa.algorithms.egwwoa import EGW_WOA

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
    p = cfg["egwwoa"]
    algo = EGW_WOA(acfg, fitness, lb, ub, rng,
                   p_levy=p["p_levy"], beta_L=p["beta_L"], b_spiral=p["b_spiral"],
                   I_stall=p["I_stall"], phi=p["phi"],
                   omega_max=p["omega"]["max"], omega_min=p["omega"]["min"])
    out = algo.run()
    print(f"Best fitness: {out['f_best']:.4f} (dim={D})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    main(ap.parse_args().config)
