using Dates
using Statistics
using LinearAlgebra
using Distributed
using BSON: @save, @load

BLAS.set_num_threads(1)

@everywhere begin
    include("src/distances.jl")
    include("src/network.jl")
    include("src/base.jl")
    include("src/io.jl")
    include("src/readLJ.jl")
end

function main()
    # Start the timer
    startTime = Dates.now()

    # Initialize the parameters
    parameters = parametersInit()

    # Initialize the input data
    inputs = inputInit(parameters)
    if parameters.mode == "training"
        confs, model, opt, refconfs, descriptorref, rng_xor = inputs
    else
        confs, model, rng_xor = inputs
    end

    println("Running MC simulation on $(nworkers()) rank(s)...\n")
    println("Starting at: ", startTime)
    println("Total number of steps: $(parameters.steps * nworkers() / 1E6)M")
    println("Number of equilibration steps per rank: $(parameters.Eqsteps / 1E6)M")
    println("Using $(parameters.paircorr) as a pair descriptor")

    if parameters.mode == "training"
        println("Using $(parameters.activation) activation")
        println("Number of iterations: $(parameters.iters)")
        println("Optimizer type: $(parameters.optimizer)")
        println("Learning rate: $(parameters.rate)")
        if parameters.optimizer == "Momentum"
            println("Momentum coefficient: $(parameters.momentum)")
        end
        # Run the training
        train!(parameters, confs, model, opt, refconfs, descriptorref, rng_xor)
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