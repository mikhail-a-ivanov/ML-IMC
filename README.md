# ML-IMC (rdfnn version)
**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

## About this version
In this version a radial distribution function, computed for a single atom,
is used as the input layer in the neural network that predicts energy contribution
of that atom.

## Features
- **Multi-reference input support:**

Use reference data from multiple systems to train a single model that works for all

- **Simulation mode:**

Run MC simulation using trained models

## How to run
Clone the repository and run:

`julia -p n ML-IMC.jl ML-IMC-init.in | tee ML-IMC.out` where `n` is the number of available cores

## Required Julia packages
**Julia version: 1.8**

Core packages:
- `Flux`
- `Distributed`
- `LinearAlgebra`
- `StaticArrays`
- `RandomNumbers`

Other packages:
- `Dates`
- `Printf`
- `BSON`
- `Statistics`

## Authors
- Prof. Alexander Lyubartsev (alexander.lyubartsev@mmk.su.se) - Principal investigator and method developer
- Maksim Posysoev (maksim.posysoev@mmk.su.se) - Software developer
- Mikhail Ivanov (mikhail.ivanov@mmk.su.se) - Software developer

## Acknowledgement
We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support, 
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
