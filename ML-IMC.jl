using Dates
using Statistics
using LinearAlgebra
using Distributed
using Chemfiles
using BSON: @save, @load

BLAS.set_num_threads(1)

@everywhere begin
    include("src/distances.jl")
    include("src/network.jl")
    include("src/base.jl")
    include("src/io.jl")
end

function main()
    # Start the timer
    startTime = Dates.now()

    # Initialize the parameters
    globalParms, MCParms, NNParms, systemParmsList = parametersInit()

    # Check if the number of workers is divisble by the number of ref systems
    @assert nworkers() % length(systemParmsList) == 0

    # Initialize the input data
    inputs = inputInit(globalParms, NNParms, systemParmsList)
    if globalParms.mode == "training"
        model, opt, refRDFs = inputs
    else
        model = inputs
    end

    println("Running MC simulation on $(nworkers()) rank(s)...\n")
    println("Starting at: ", startTime)
    println("Total number of steps: $(MCParms.steps * nworkers() / 1E6)M")
    println("Number of equilibration steps per rank: $(MCParms.Eqsteps / 1E6)M")
    #println("Neural network architecture: $(NNParms.neurons)")

    if globalParms.mode == "training"
        println("Using $(NNParms.activation) activation")
        println("Number of iterations: $(NNParms.iters)")
        println("Optimizer type: $(NNParms.optimizer)")
        println("Learning rate: $(NNParms.rate)")
        if NNParms.optimizer == "Momentum"
            println("Momentum coefficient: $(NNParms.μ)")
        end
        
        # Run the training
        train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    end

    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    println("Walltime: ", wallTime)
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end