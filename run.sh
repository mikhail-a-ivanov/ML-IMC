#!/bin/bash

folder_name=$(basename "$PWD")
num_processes=4

screen -S "$folder_name" -dm bash -c "julia --project=. -p $num_processes src/main.jl configs/methanol-data/config.toml | tee report.out"