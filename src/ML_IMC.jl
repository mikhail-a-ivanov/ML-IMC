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
    println("Start time: ", Dates.format(start_time, "dd u yyyy, HH:MM"))

    println("Initialization")
    println("--------------")

    # Initialize parameters and validate configuration
    global_params, mc_params, nn_params, pretrain_params, system_params_list = parameters_init()

    log_global_parameters(global_params)

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

    # Log configuration information
    log_symmetry_functions_info(nn_params)
    log_model_info(model, nn_params)
    log_training_config(global_params, mc_params)

    # Execute workflow based on mode
    if global_params.mode == "training"
        # Execute pretraining if needed
        if global_params.model_file == "none"
            log_optimizer_info(optimizer)
            model = pretrain_model!(pretrain_params, nn_params, system_params_list, model, optimizer, ref_rdfs)

            println("\nOptimizer for the training")
            optimizer = init_optimizer(nn_params)
        end
        log_optimizer_info(optimizer)

        train!(global_params, mc_params, nn_params, system_params_list, model, optimizer, ref_rdfs)
    else
        length(system_params_list) == 1 || throw(ArgumentError("Simulation mode supports only one system"))
        simulate!(model, global_params, mc_params, nn_params, system_params_list[1])
    end

    # Log execution summary
    stop_time = now()
    wall_time = canonicalize(stop_time - start_time)
    println("\nExecution completed:")
    println("Stopped time: ", Dates.format(stop_time, "dd u yyyy, HH:MM"))
    println("Wall time: ", wall_time)

    # Clean up after execution
    GC.gc()
    exit(0)
end
end # module ML_IMC
