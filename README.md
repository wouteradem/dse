# Dyson-Schwinger Equation Solver

This project provides a C++ implementation for solving Dyson-Schwinger equations in 0-dimensional quantum field theory, based on the methods described in the paper:

**[Dyson-Schwinger equations in zero dimensions and polynomial approximations](https://arxiv.org/abs/2307.01008v1)**

## Compilation

You need a C++17-compliant compiler. This project was developed and tested on **macOS**.

### For the ϕ⁴-theory:
```bash
g++ dse.cpp -o dse -std=c++17
```

### For the iϕ³-theory
```bash
g++ dse_i.cpp -o dse_i -std=c++17
```

### Program Features
- ```dse.cpp``` solves the Dyson-Schwinger equation for the ϕ⁴-theory.
- ```dse_i.cpp``` solves it for the iϕ³-theory.
- Each program allows two modes: Hard truncation or Asymptotic completion
- The resulting Green's function equation is solved numerically, and the roots (zeros) are exported to a CSV file.

### Output
```roots.csv```

### Visualization
The ```plots/``` directory contains Python scripts to visualize the roots.

```bash
cd plots

# For ϕ⁴-theory:
python3 plot_phi4.py

# For iϕ³-theory:
python3 plot_iphi3.py
```