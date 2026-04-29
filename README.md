# ML-IMC

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

This project implements an algorithm for creating neural network force fields and performing simulations.

## Installation

1. **Julia Version:**  Ensure you have Julia version 1.10 or later installed.

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

### Required fields per mode

| Field | training | pmf-pretraining | magic-pretraining | simulation |
|---|---|---|---|---|
| `[global].model_file` | "none" or path | "none" or path | "none" or path | **required** (path to .bson) |
| `[global].system_files` | 1+ systems | 1+ systems | 1+ systems | **exactly 1** system |
| `[pretraining]` section | not used | required | required | not used |
| `[magic_pretraining].potential_files` | not used | not used | **required** (1 per system) | not used |

## Configuration Files

- **`config.toml`:**  Main configuration with `[global]`, `[monte_carlo]`, `[neural_network]`, `[pretraining]`, and `[magic_pretraining]` sections. See `configs/methanol-data/config.toml.example` for a minimal reproducible example.
- **`symmetry_functions.toml`:**  Defines Behler-Parrinello symmetry functions (G2/G3/G9) used as input features. Set `use_g3 = true` / `use_g9 = true` to enable angular functions.
- **System `*.toml` files:**  One per system in `configs/methanol-data/`, specifying PDB topology, XTC trajectory, reference RDF, and simulation parameters.

### Magic Potential Format

Magic pre-training potential files (`.dat`/`.pot`) contain two whitespace-separated columns: `r` (distance in Å) and `U(r)` (pair potential). See `scripts/prepare_pot.py` for preprocessing IMC potentials into the expected format.

## Output

All output files are written to `global.output_dir` (default: `run/`). This includes models (`.bson`), optimizer states, RDF predictions, energy time series, and training/pre-training logs.

## Authors

- Prof. Alexander Lyubartsev (<alexander.lyubartsev@su.se>) - Principal investigator and method developer
- Mikhail Ivanov (<mikhail.ivanov@su.se>) - Software developer
- Maksim Posysoev (<maksim.posysoev@su.se>) - Software developer

## Acknowledgement

We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support,
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
