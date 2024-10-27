using BSON: @load
include("entities.jl")
include("utils.jl")
include("logging.jl")
include("neural_network.jl")

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
        # Initialize the model
        println("Initializing a new neural network with random weights")

        model = model_init(nn_params)

        print_model_summary(model, nn_params)

        if global_params.optimizer_file != "none"
            println("Ignoring given optimizer filename...")
        end
        if global_params.gradients_file != "none"
            println("Ignoring given gradients filename...")
        end
        # Run pre-training if no initial model is given
        opt = init_optimizer(pretrain_params)
        # Restart the training
    else
        # Loading the model
        check_file(global_params.model_file)
        println("Reading model from $(global_params.model_file)")
        @load global_params.model_file model

        if global_params.mode == "training"
            # Either initialize the optimizer or read from a file
            if global_params.optimizer_file != "none"
                check_file(global_params.optimizer_file)
                println("Reading optimizer state from $(global_params.optimizer_file)")
                @load global_params.optimizer_file opt
            else
                opt = init_optimizer(nn_params)
            end

            mean_loss_gradients = nothing
            # Optionally read gradients from a file
            if global_params.gradients_file != "none"
                check_file(global_params.gradients_file)
                println("Reading gradients from $(global_params.gradients_file)")
                @load global_params.gradients_file mean_loss_gradients
            end

            # Update the model if both opt and gradients are restored
            if global_params.optimizer_file != "none" && global_params.gradients_file != "none"
                println("\nUsing the restored gradients and optimizer to update the current model...\n")
                update_model!(model, opt, mean_loss_gradients)

                # Skip updating if no gradients are provided
            elseif global_params.optimizer_file != "none" && global_params.gradients_file == "none"
                println("\nNo gradients were provided, rerunning the training iteration with the current model and restored optimizer...\n")

                # Update the model if gradients are provided without the optimizer:
                # valid for optimizer that do not save their state, e.g. Descent,
                # otherwise might produce unexpected results
            elseif global_params.optimizer_file == "none" && global_params.gradients_file != "none"
                println("\nUsing the restored gradients with reinitialized optimizer to update the current model...\n")
                update_model!(model, opt, mean_loss_gradients)
            else
                println("\nNeither gradients nor optimizer were provided, rerunning the training iteration with the current model...\n")
            end
        end
    end

    if global_params.mode == "training"
        return (model, opt, ref_rdfs)
    else
        return (model)
    end
end
