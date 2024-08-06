# ML-IMC

**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

This project implements an algorithm for creating neural network force fields and performing simulations.

## Installation

1. **Julia Version:**  Ensure you have Julia version 1.10 or later installed.

2. **Package Installation:** Open the Julia REPL and install the required packages:

  ```julia
   using Pkg
   Pkg.add(["Distributed", "LinearAlgebra", "Dates", "Flux", "Statistics", "BSON", "RandomNumbers", "Chemfiles", "Printf"])
  ```

## Running the Code

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mikhail-a-ivanov/ML-IMC.git
   ```

2. **Configuration:**
   - Modify the `ML-IMC-init.in` file to specify the desired parameters for training or simulation.
   - See the detailed parameter descriptions in the `ML-IMC-init.in` file.

3. **Execution:**
   - **Training:**

     ```bash
     julia -p n ML-IMC.jl ML-IMC-init.in | tee report.out 
     ```

     where `n` is the number of systems you want to train.

   - **Simulation:**
     - Set the `mode` parameter in `ML-IMC-init.in` to "simulation".
     - Specify the input PDB file and trained model file in the appropriate parameters.
     - Run the same command as for training.

## Input Files

- **`ML-IMC-init.in`:**  Main input file containing global parameters, Monte Carlo settings, neural network parameters, and pre-training parameters.
- **`symmetry-functions.in`:**  Defines the symmetry functions to be used as input features for the neural network.
- **System Input Files (`.in`):**  Individual input files for each system to be trained, containing information such as topology, reference RDF, and simulation parameters.

## Authors

- Prof. Alexander Lyubartsev (<alexander.lyubartsev@mmk.su.se>) - Principal investigator and method developer
- Mikhail Ivanov (<mikhail.ivanov@mmk.su.se>) - Software developer
- Maksim Posysoev (<maksim.posysoev@mmk.su.se>) - Software developer

## Acknowledgement

We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support,
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
