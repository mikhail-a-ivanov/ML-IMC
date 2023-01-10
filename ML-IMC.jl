using Distributed
using LinearAlgebra
using Dates

BLAS.set_num_threads(1)

@everywhere begin
    include("src/distances.jl")
    include("src/network.jl")
    include("src/base.jl")
    include("src/io.jl")
    include("src/pretraining.jl")
end

function main()
    # Start the timer
    startTime = Dates.now()
    println("Starting at: ", startTime)

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

    println("Using $(NNParms.neurons[1]) G2 values as the neural input for each atom")
    η = 1 / (2 * NNParms.sigma^2)
    println(
        "G2 sigma parameter: $(NNParms.sigma) Å; the corresponding eta parameter: $(round(η, digits=2)) Å^-2",
    )
    Rss = Array{Float64}(LinRange(NNParms.minR, NNParms.maxR, NNParms.neurons[1]))
    println("G2 Rs values: $(round.(Rss, digits=2)) Å")
    println("G2 cutoff radius: $(NNParms.maxR) Å")

    if globalParms.mode == "training"
        nsystems = length(systemParmsList)
        println("Training a model using $(nsystems) reference system(s)")
        if globalParms.inputmodel == "random"
            println("Initializing a new neural network with random weigths")
        elseif globalParms.inputmodel == "zero"
            println(
                "Initializing a new neural network with zero weigths in the first layer",
            )
        else
            println("Starting training from $(globalParms.inputmodel)")
        end
        println("Using the following activation functions: $(NNParms.activations)")
        println("Neural network regularization parameter: $(NNParms.REGP)")
        println("Number of iterations: $(NNParms.iters)")
        println("Optimizer type: $(NNParms.optimizer)")
        println("Parameters of optimizer:")
        if NNParms.optimizer == "Momentum"
            println("Learning rate: $(NNParms.rate)")
            println("Momentum coefficient: $(NNParms.momentum)")

        elseif NNParms.optimizer == "Descent"
            println("Learning rate: $(NNParms.rate)")

        elseif NNParms.optimizer == "Nesterov"
            println("Learning rate: $(NNParms.rate)")
            println("Momentum coefficient: $(NNParms.momentum)")

        elseif NNParms.optimizer == "RMSProp"
            println("Learning rate: $(NNParms.rate)")
            println("Momentum coefficient: $(NNParms.momentum)")

        elseif NNParms.optimizer == "Adam"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "RAdam"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "AdaMax"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "AdaGrad"
            println("Learning rate: $(NNParms.rate)")

        elseif NNParms.optimizer == "AdaDelta"
            println("Learning rate: $(NNParms.rate)")

        elseif NNParms.optimizer == "AMSGrad"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "NAdam"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "AdamW"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "OAdam"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")

        elseif NNParms.optimizer == "AdaBelief"
            println("Learning rate: $(NNParms.rate)")
            println("Decay 1: $(NNParms.decay1)")
            println("Decay 2: $(NNParms.decay2)")
        end

        # Run pretraining
        model = preTrain!(NNParms, systemParmsList, model, opt, refRDFs)
        # Run the training
        println("Running MC simulation on $(nworkers()) rank(s)...\n")
        println("Total number of steps: $(MCParms.steps * nworkers() / 1E6)M")
        println("Number of equilibration steps per rank: $(MCParms.Eqsteps / 1E6)M")
        train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    else
        @assert length(systemParmsList) == 1
        println("Running simulation with $(globalParms.inputmodel)")
        # Run the simulation
        simulate!(model, globalParms, MCParms, NNParms, systemParmsList[1])
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
