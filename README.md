# ML-IMC
**M**achine **L**earning enhanced **I**nverse **M**onte **C**arlo.

## How to run
Clone the repository and run:

`julia -p n ML-IMC.jl ML-IMC-init.in | tee mc.out` where `n` is the number of available cores

## Required Julia packages
**Julia version: 1.8**

Core packages:
- `Flux`
- `Distributed`
- `Chemfiles`
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
- Mikhail Ivanov (mikhail.ivanov@mmk.su.se) - Software developer
- Maksim Posysoev (maksim.posysoev@mmk.su.se) - Software developer

## Acknowledgement
We would like to thank the Åke Åkesons foundation as well as Swedish Research Council (Vetenskapsrådet) for the financial support, 
and Swedish National Infrastructure for Computing (SNIC) for providing the high-performance computing systems.
