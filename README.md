# ML-IMC
**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

## Planned features
- Add distance histograms with gaussian decay as an option for training
- Add the option to use additional types of reference data (angle distributions, etc.)
- Introduce training with multiple reference data sets (obtained for different temperatures, densities or concentrations)
- Introduce `training` and `simulation` modes for ML-IMC
- Implement auto-adjustment of MC step length

## How to run
Clone the repository and run:

`julia -p n ML-IMC.jl ML-IMC-init.in > ML-IMC.out` where `n` is the number of available cores

## Required Julia packages
**Julia version: 1.7**

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
- Mikhail Ivanov (mikhail.ivanov@mmk.su.se) - Main software developer

## Acknowledgement
We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support, 
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
