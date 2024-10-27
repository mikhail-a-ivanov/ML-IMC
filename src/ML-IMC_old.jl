using Distributed
using LinearAlgebra
using Dates

BLAS.set_num_threads(1)

@everywhere begin
    include("io.jl")
    include("distances.jl")
    include("optimizer.jl")
    include("network.jl")
    include("symmfunctions.jl")
    include("base.jl")
    include("pretraining.jl")
end

function main()
    # Start the timer
    startTime = Dates.now()
    println("Starting at: ", startTime)

    # Initialize the parameters
    globalParms, MCParms, NNParms, preTrainParms, systemParmsList = parametersInit()

    # Check if the number of workers is divisble by the number of ref systems
    num_workers = nworkers()
    num_systems = length(systemParmsList)
    @assert(num_workers % num_systems==0,
            "Number of requested CPU cores ($num_workers) "*"must be divisible by the number of systems ($num_systems)!")

    # Initialize the input data
    inputs = inputInit(globalParms, NNParms, preTrainParms, systemParmsList)

    if globalParms.mode == "training"
        model, opt, refRDFs = inputs
    else
        model = inputs
    end

    # Print information about symmetry functions
    println("Using the following symmetry functions as the neural input for each atom:")
    print_symmetry_function_info(NNParms)

    if globalParms.mode == "training"
        println("Training a model using $(num_systems) reference system(s)")
        println("Using the following activation functions: $(NNParms.activations)")
        if globalParms.modelFile == "none"
            # Run pretraining
            model = pretrain!(preTrainParms, NNParms, systemParmsList, model, opt, refRDFs)
            # Restore optimizer state to default
            println("\nRe-initializing the optimizer for the training...\n")
            opt = optInit(NNParms)
            reportOpt(opt)
            println("Neural network regularization parameter: $(NNParms.REGP)")
        end
        # Run the training
        println("\nStarting the main part of the training...\n")
        println("Adaptive gradient scaling: $(globalParms.adaptiveScaling)")
        println("Number of iterations: $(NNParms.iters)")
        println("Running MC simulation on $(num_workers) rank(s)...\n")
        println("Total number of steps: $(MCParms.steps * num_workers / 1E6)M")
        println("Number of equilibration steps per rank: $(MCParms.Eqsteps / 1E6)M")
        train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    else
        @assert(length(systemParmsList)==1, "Only one system at a time can be simulated!")
        println("Running simulation with $(globalParms.modelFile)")
        # Run the simulation
        simulate!(model, globalParms, MCParms, NNParms, systemParmsList[1])
    end

    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    return println("Walltime: ", wallTime)
end

function print_symmetry_function_info(NNParms)
    if !isempty(NNParms.G2Functions)
        println("    G2 symmetry functions:")
        println("    eta, Å^-2; rcutoff, Å; rshift, Å")
        for G2Function in NNParms.G2Functions
            println("       ", G2Function)
        end
    end
    if !isempty(NNParms.G3Functions)
        println("    G3 symmetry functions:")
        println("    eta, Å^-2; lambda; zeta; rcutoff, Å; rshift, Å")
        for G3Function in NNParms.G3Functions
            println("       ", G3Function)
        end
    end
    if !isempty(NNParms.G9Functions)
        println("    G9 symmetry functions:")
        println("    eta, Å^-2; lambda; zeta; rcutoff, Å; rshift, Å")
        for G9Function in NNParms.G9Functions
            println("       ", G9Function)
        end
    end
    println("Maximum cutoff distance: $(NNParms.maxDistanceCutoff) Å")
    return println("Symmetry function scaling parameter: $(NNParms.symmFunctionScaling)")
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
