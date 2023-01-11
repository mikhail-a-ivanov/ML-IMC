
# Run this module in REPL to get access to the input variables
include("src/distances.jl")
include("src/network.jl")
include("src/base.jl")
include("src/io.jl")
include("src/pretraining-mc.jl")

globalParms, MCParms, NNParms, systemParmsList = parametersInit()

inputs = inputInit(globalParms, NNParms, systemParmsList)

model, opt, refRDFs = inputs
