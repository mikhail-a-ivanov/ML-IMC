
# Start the REPL with `julia -p n`
# Run this module in REPL to get access to the input variables
using Distributed
using LinearAlgebra
using Dates

using Plots
using BenchmarkTools

BLAS.set_num_threads(1)

@everywhere begin
    include("src/distances.jl")
    include("src/network.jl")
    include("src/base.jl")
    include("src/io.jl")
    include("src/pretraining.jl")
end

# Initialize the parameters
globalParms, MCParms, NNParms, preTrainParms, systemParmsList = parametersInit()

@assert nworkers() % length(systemParmsList) == 0

# Initialize the input data
inputs = inputInit(globalParms, NNParms, preTrainParms, systemParmsList)
if globalParms.mode == "training"
    model, opt, refRDFs = inputs
else
    model = inputs
end
