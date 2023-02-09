#!/bin/bash

# Create as many folders as you have CPU cores
# Single core ML-IMC run will start in every such folder
for run_folder in run*
do
 cp -r src ./$run_folder
 cp *jl ./$run_folder
 cp 100CH3OH-CG.in ML-IMC-init.in symmetry-functions.in ./$run_folder
 echo "Starting" $run_folder...
 cd $run_folder
 julia ML-IMC.jl > mc.out &
 cd ../
done

