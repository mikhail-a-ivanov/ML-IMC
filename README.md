# ML-IMC

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

This project implements an algorithm for creating neural network force fields and performing simulations.

## Installation

1. **Julia Version:**  Ensure you have Julia version 1.11 or later installed.

2. **Package Installation:**

   ```bash
   make install
   ```
   or
   ```julia
   using Pkg; Pkg.instantiate()
   ```

## Operation Modes

The project supports 4 independent modes, selected via `mode` in the `[global]` section of `config.toml`:

| Mode | Description |
|---|---|
| `training` | Main ML-IMC training loop. Iteratively samples MC configurations, computes IMC gradients, and updates the neural network potential. |
| `pmf-pretraining` | Pre-trains the neural network to match Potential of Mean Force (PMF) derived from reference RDF data. |
| `magic-pretraining` | Pre-trains the neural network on externally obtained IMC potentials (pair potentials from previous IMC calculations). Requires matching `.dat`/`.pot` files. |
| `simulation` | Runs a production MC simulation with an already trained model. Requires exactly one system and a valid `model_file`. |

No mode requires commenting/uncommenting code or manually editing Julia files. All mode-specific behavior is configured in `config.toml`.

## Running

```bash
julia -p N src/main.jl configs/methanol-data/config.toml | tee report.out
```

Where `N` is the number of worker processes (must be divisible by the number of systems).

For `simulation` mode, use exactly one system and set `N = 1`.

## Authors

- Prof. Alexander Lyubartsev (<alexander.lyubartsev@su.se>) - Principal investigator and method developer
- Mikhail Ivanov (<mikhail.ivanov@su.se>) - Software developer
- Maksim Posysoev (<maksim.posysoev@su.se>) - Software developer

## Acknowledgement

We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support,
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
