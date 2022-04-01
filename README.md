# ML-IMC
**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

## Planned features
- Use radial distribution functions or distance histograms with gaussian decay for training instead of plain distance histograms
- Add the option to use additional types of reference data (angle distributions, etc.)
- Introduce training with multiple reference data sets (obtained for different temperatures, densities or concentrations)
- Add support for multilayered neural networks and different types of activation functions

## How to run
Clone the repository and run:

`julia -p 40 ML-IMC-run.jl LJML-init.in > p40.out`

## Required Julia packages
**Julia version: 1.7**

Core packages:
- Flux
- Distributed
- LinearAlgebra
- StaticArrays
- RandomNumbers

Other packages:
- Dates
- Printf
- BSON
- Statistics

## Authors
- Prof. Alexander Lyubartsev (alexander.lyubartsev@mmk.su.se) - Principal investigator and method developer
- Mikhail Ivanov (mikhail.ivanov@mmk.su.se) - Main software developer

## Acknowledgement
We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support, 
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
