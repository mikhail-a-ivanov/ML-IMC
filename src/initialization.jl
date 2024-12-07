using ..ML_IMC

function input_init(global_params::GlobalParameters, nn_params::NeuralNetParameters,
                    pretrain_params::PreTrainingParameters, system_params_list::Vector{SystemParameters})
    # Read reference data
    ref_rdfs = []
    for system_params in system_params_list
        bins, ref_rdf = read_rdf(system_params.rdf_file)
        append!(ref_rdfs, [ref_rdf])
    end

    # Initialize the model and the optimizer
    if global_params.model_file == "none"
        model = model_init(nn_params)
        opt = init_optimizer(pretrain_params)
        optimizer = Flux.setup(opt, model)
    else
        # Loading the model
        check_file(global_params.model_file)
        @load global_params.model_file model

        if global_params.mode == "training"
            # Either initialize the optimizer or read from a file
            if global_params.optimizer_file != "none"
                check_file(global_params.optimizer_file)
                @load global_params.optimizer_file optimizer
            else
                opt = init_optimizer(nn_params)
                optimizer = Flux.setup(opt, model)
            end

            mean_loss_gradients = nothing
            # Optionally read gradients from a file
            if global_params.gradients_file != "none"
                check_file(global_params.gradients_file)
                @load global_params.gradients_file mean_loss_gradients
            end

            # Update the model if both opt and gradients are restored
            if global_params.optimizer_file != "none" && global_params.gradients_file != "none"
                update_model!(model, optimizer, mean_loss_gradients)
            end
        end
    end

    if global_params.mode == "training"
        return (model, optimizer, ref_rdfs)
    else
        return (model)
    end
end
