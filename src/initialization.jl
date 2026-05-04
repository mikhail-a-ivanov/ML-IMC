using ..ML_IMC

function input_init(global_params::GlobalParameters, nn_params::NeuralNetParameters,
                    pretrain_params::PreTrainingParameters, magic_params::MagicPreTrainingParameters,
                    system_params_list::Vector{SystemParameters})
    # Read RDF targets used by training and pretraining losses.
    rdf_targets = [read_rdf(system_params.rdf_file)[2] for system_params in system_params_list]

    mode = global_params.mode
    model = nothing

    model_file = global_params.model_file

    # Initialize or load model
    model = if model_file == "none"
        model_init(nn_params)
    else
        check_file(model_file)
        data = BSON.load(model_file)

        # Validate model key
        if !haskey(data, :model)
            throw(ArgumentError("Loaded BSON file $model_file does not contain a 'model' variable."))
        end

        loaded_model = data[:model]

        # Validate type
        if !(loaded_model isa Flux.Chain)
            throw(ArgumentError("Loaded object 'model' from $model_file is not a Flux.Chain (got $(typeof(loaded_model)))."))
        end

        # Validate input dimension
        n_g2 = length(nn_params.g2_functions)
        first_layer = loaded_model[1]
        if hasfield(typeof(first_layer), :weight)
            n_inputs = size(first_layer.weight, 2)
            if n_inputs != n_g2
                throw(ArgumentError("Model in $model_file has $n_inputs inputs, but current config specifies $n_g2 symmetry functions. " *
                                    "Check your configuration and symmetry function file."))
            end
        else
            @warn "Could not verify model input dimension: first layer $(typeof(first_layer)) does not have a weight field."
        end

        f32(loaded_model)
    end

    # For simulation, model is mandatory
    if mode == "simulation" && model_file == "none"
        throw(ArgumentError("Simulation mode requires a model file. Set checkpoint.model_file to a valid .bson path."))
    end

    # For simulation, no optimizer needed
    if mode == "simulation"
        return (model, nothing, nothing, rdf_targets)
    end

    # Initialize or load optimizer
    opt_state = nothing
    optimizer_file = global_params.optimizer_file

    if optimizer_file != "none"
        check_file(optimizer_file)
        data = BSON.load(optimizer_file)

        # Validate optimizer state key
        if !haskey(data, :opt_state)
            throw(ArgumentError("Loaded BSON file $optimizer_file does not contain 'opt_state' variable."))
        end

        opt_state = f32(data[:opt_state])
    end

    # Choose optimizer based on mode
    if mode == "training"
        optimizer = init_optimizer(nn_params)
    else
        optimizer = init_optimizer(pretrain_params)
    end

    return (model, optimizer, opt_state, rdf_targets)
end
