using ..ML_IMC

function input_init(global_params::GlobalParameters, nn_params::NeuralNetParameters,
                    pretrain_params::PreTrainingParameters, system_params_list::Vector{SystemParameters})
    # Read reference data
    ref_rdfs = [read_rdf(system_params.rdf_file)[2] for system_params in system_params_list]

    model = nothing

    # Early return for non-training mode with existing model
    if global_params.mode != "training" && global_params.model_file != "none"
        check_file(global_params.model_file)
        @load global_params.model_file model
        return (model, nothing, nothing, ref_rdfs)
    end

    # Initialize or load model
    model = if global_params.model_file == "none"
        model_init(nn_params)
    else
        check_file(global_params.model_file)
        @load global_params.model_file model
        model
    end

    # Return early if not in training mode
    if global_params.mode != "training"
        return (model, nothing, nothing, ref_rdfs)
    end

    # Initialize or load optimizer
    opt_state = nothing
    optimizer = if global_params.optimizer_file != "none"
        check_file(global_params.optimizer_file)
        @load global_params.optimizer_file opt_state
        init_optimizer(nn_params)
    else
        global_params.model_file == "none" ? init_optimizer(pretrain_params) : init_optimizer(nn_params)
    end

    return (model, optimizer, opt_state, ref_rdfs)
end
