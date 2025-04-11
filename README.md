# Hybrid GW-WOA Algorithm for Renewable Energy Storage Optimization

This repository contains a fully implemented and reproducible framework for optimizing renewable energy storage using a hybrid metaheuristic approach. The method, **GW-WOA with Lévy flights and chaotic mechanisms**, integrates the global exploration capabilities of Grey Wolf Optimization (GWO) with the local exploitation efficiency of Whale Optimization Algorithm (WOA), further enhanced by dynamic weights, chaotic escape, and Lévy-based search jumps.

## 🔬 Academic Purpose

This repository and its contents were developed by **Ercan Erkalkan** (Marmara University) solely for academic research and publication purposes. The work supports a peer-reviewed submission to a high-impact journal in the field of sustainable computing and energy optimization.

## 🧠 Key Features

- Hybridization of GWO and WOA with dynamic behavior.
- Integration of **Lévy flight** and **chaotic logistic reinitialization** mechanisms.
- Adaptable for multi-objective and constraint-based optimization tasks.
- Supports comparative evaluation with PSO, GA, HS, and FPA algorithms.
- Synthetic yet realistic data modeling for solar/wind power and grid demand fluctuations.

## 📁 Repository Structure

```bash
.
├── gwwo.py               # Core implementation of GW-WOA, GWO, WOA, HS, FPA classes
├── renewable_optimizer.py  # Application logic, fitness functions, comparisons
├── gwwo.cpython-311.pyc  # Compiled Python object (optional)
├── average_convergence.png
├── soc_comparison.png
├── soc_comparison_subplots.png
├── population_sensitivity.png
```

## 📊 Performance Results

The hybrid algorithm was benchmarked over 100 independent trials against four standard metaheuristics (WOA, CPSO, HS, FPA). Highlights include:

- **15.48% cost reduction** over standard WOA.
- **79.93% performance gain** over classical Harmony Search (HS).
- Consistent convergence behavior and lowest standard deviation among all tested algorithms.

All performance plots are included in this repository.

## 📄 License

This project is licensed under the terms of the MIT license.

```text
MIT License

Copyright (c) 2025 Ercan Erkalkan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND...
```

## 🔍 Source Code Availability Statement

All source codes, simulations, and analysis tools developed and utilized in this study are publicly available in the following GitHub repository:

➡️ **[https://github.com/hatikuay/improved_gw_woa_levy](https://github.com/hatikuay/improved_gw_woa_levy)**

This ensures full transparency and reproducibility of our optimization experiments and comparative evaluations.

## 📫 Contact

For academic correspondence:

- **Name:** Öğr. Gör. Dr. Ercan ERKALKAN  
- **Institution:** Marmara University, Vocational School of Technical Sciences  
- **Email:** ercan.erkalkan@marmara.edu.tr  
- **Web:** [https://avesis.marmara.edu.tr/ercan.erkalkan](https://avesis.marmara.edu.tr/ercan.erkalkan)
