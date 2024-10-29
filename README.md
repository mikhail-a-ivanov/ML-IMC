# ML-IMC

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

This project implements an algorithm for creating neural network force fields and performing simulations.

## Installation

1. **Julia Version:**  Ensure you have Julia version 1.10 or later installed.

2. **Package Installation:** Open the Julia REPL and install the required packages:

  ```julia
   using Pkg
   Pkg.instantiate()
  ```

  or

  ```bash
  make install
  ```

## Running the Code

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mposysoev/ML-IMC --depth=1
   ```

2. **Configuration:**
   - Modify the `config.toml` file (example in the folder `configs`) to specify the desired parameters for training or simulation.
   - Modify the `symmetry_functions.toml` file for specifying trajectory descriptors.

3. **Execution:**
   - **Training:**

     ```bash
     julia -p n src/main.jl configs/config.toml | tee report.out 
     ```

     where `n` is the number of systems you want to train.

   - **Simulation:**
     - Set the `mode` parameter in `config.toml` to "simulation".
     - Specify the input PDB file and trained model file in the appropriate parameters.
     - Run the same command as for training.

## Input Files

- **`config.toml`:**  Main input file containing Global parameters, Monte Carlo settings, Neural Network parameters, and Pre-training parameters.
- **`symmetry_functions.toml`:**  Defines the symmetry functions to be used as input features for the neural network.
- **System Input Files (`.toml`):**  Individual input files for each system to be trained, containing information such as topology, reference RDF, and simulation parameters.

## Authors

- Prof. Alexander Lyubartsev (<alexander.lyubartsev@mmk.su.se>) - Principal investigator and method developer
- Mikhail Ivanov (<mikhail.ivanov@mmk.su.se>) - Software developer
- Maksim Posysoev (<maksim.posysoev@mmk.su.se>) - Software developer

## Acknowledgement

We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support,
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
