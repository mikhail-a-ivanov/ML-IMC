#!/bin/bash -l
#SBATCH -J mlimc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -e mc.err

julia ML-IMC.jl | tee mc.out
