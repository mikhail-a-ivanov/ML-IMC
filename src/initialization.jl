using ..ML_IMC

function input_init(global_params::GlobalParameters, nn_params::NeuralNetParameters,
                    pretrain_params::PreTrainingParameters, magic_params::MagicPreTrainingParameters,
                    system_params_list::Vector{SystemParameters})
    # Read reference data
    ref_rdfs = [read_rdf(system_params.rdf_file)[2] for system_params in system_params_list]

    mode = global_params.mode
    model = nothing

    model_file = global_params.model_file

    # Initialize or load model
    model = if model_file == "none"
        model_init(nn_params)
    else
        check_file(model_file)
        @load model_file model
        model
    end

    # For simulation, model is mandatory
    if mode == "simulation" && model_file == "none"
        throw(ArgumentError("Simulation mode requires a model file. Set checkpoint.model_file to a valid .bson path."))
    end

    # For simulation, no optimizer needed
    if mode == "simulation"
        return (model, nothing, nothing, ref_rdfs)
    end

    # Initialize or load optimizer
    opt_state = nothing
    optimizer_file = global_params.optimizer_file

    if optimizer_file != "none"
        check_file(optimizer_file)
        @load optimizer_file opt_state
    end

    # Choose optimizer based on mode
    if mode == "training"
        optimizer = init_optimizer(nn_params)
    else
        optimizer = init_optimizer(pretrain_params)
    end

    return (model, optimizer, opt_state, ref_rdfs)
end
