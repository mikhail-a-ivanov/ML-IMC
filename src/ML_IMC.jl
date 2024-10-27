module ML_IMC

__precompile__()

using Dates
using BSON: @load, @save
using Flux
using Statistics
using RandomNumbers
using RandomNumbers.Xorshifts
using TOML
using Chemfiles
using LinearAlgebra
using Printf
using Distributed

include("entities.jl")
include("config_loading.jl")
include("distances.jl")
include("gradients.jl")
include("initialization.jl")
include("logging.jl")
include("monte_carlo.jl")
include("neural_network.jl")
include("pre_training.jl")
include("simulation.jl")
include("symmetry_functions.jl")
include("training.jl")
include("utils.jl")

BLAS.set_num_threads(1)

function main()
    # Initialize timing
    start_time = now()
    println("Starting at: ", start_time)

    # Initialize parameters and validate configuration
    global_params, mc_params, nn_params, pretrain_params, system_params_list = parameters_init()

    # Validate worker/system configuration
    num_workers = nworkers()
    num_systems = length(system_params_list)

    if num_workers % num_systems != 0
        throw(ArgumentError("Number of workers ($num_workers) must be divisible by number of systems ($num_systems)"))
    end

    # Initialize input data and model components
    inputs = input_init(global_params, nn_params, pretrain_params, system_params_list)
    model, optimizer, ref_rdfs = if global_params.mode == "training"
        inputs
    else
        inputs, nothing, nothing
    end

    # Display model configuration
    println("Using the following symmetry functions as the neural input for each atom:")
    print_symmetry_function_info(nn_params)

    # Execute workflow based on mode
    if global_params.mode == "training"
        println("""
            Training Configuration:
            - Using $(num_systems) reference system(s)
            - Activation functions: $(nn_params.activations)
            """)

        # Execute pretraining if needed
        if global_params.model_file == "none"
            model = pretrain_model!(pretrain_params, nn_params, system_params_list, model, optimizer, ref_rdfs)

            println("\nRe-initializing the optimizer for the training...")
            optimizer = init_optimizer(nn_params)
            report_optimizer(optimizer)
            println("Neural network regularization parameter: $(nn_params.regularization)")
        end

        # Execute main training
        println("""
            \nStarting main training phase:
            - Adaptive gradient scaling: $(global_params.adaptive_scaling)
            - Iterations: $(nn_params.iterations)
            - Running on $(num_workers) worker(s)
            - Total steps: $(mc_params.steps * num_workers / 1e6)M
            - Equilibration steps per rank: $(mc_params.equilibration_steps / 1e6)M
            """)

        train!(global_params, mc_params, nn_params, system_params_list, model, optimizer, ref_rdfs)
    else
        length(system_params_list) == 1 || throw(ArgumentError("Simulation mode supports only one system"))
        println("Running simulation with $(global_params.model_file)")
        simulate!(model, global_params, mc_params, nn_params, system_params_list[1])
    end

    # Log execution summary
    stop_time = now()
    wall_time = canonicalize(stop_time - start_time)
    println("\nExecution completed:")
    println("- Stop time: ", stop_time)
    println("- Wall time: ", wall_time)
end

end # module ML_IMC
