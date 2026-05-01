using ..ML_IMC
using Dates

function compute_training_loss(descriptor_nn::AbstractVector{T},
                               descriptor_ref::AbstractVector{T},
                               model::Flux.Chain,
                               nn_params::NeuralNetParameters) where {T <: AbstractFloat}
    descriptor_loss_rmse = sqrt(mean(abs2, descriptor_nn .- descriptor_ref))
    descriptor_loss_mae = mean(abs, descriptor_nn .- descriptor_ref)

    reg_loss = zero(T)
    if nn_params.regularization > zero(T)
        reg_loss = nn_params.regularization * sum(sum(abs2, p) for p in Flux.trainables(model))
    end

    total_loss_rmse = descriptor_loss_rmse + reg_loss
    total_loss_mae = descriptor_loss_mae + reg_loss

    return total_loss_mae, total_loss_rmse
end

function prepare_monte_carlo_inputs(global_params::GlobalParameters,
                                    mc_params::MonteCarloParameters,
                                    nn_params::NeuralNetParameters,
                                    system_params_list::Vector{SystemParameters},
                                    model::Flux.Chain)
    n_systems = length(system_params_list)
    n_workers = nworkers()

    # Validate that workers can be evenly distributed across systems
    if n_workers % n_systems != 0
        throw(ArgumentError("Number of workers ($n_workers) must be divisible by number of systems ($n_systems)"))
    end

    # Get model weights for serialization
    model_weights, _ = Flux.destructure(model)
    model_weights = Float32.(model_weights)

    # Create input for each system
    reference_inputs = Vector{MonteCarloSampleInput}(undef, n_systems)
    for (i, system_params) in enumerate(system_params_list)
        reference_inputs[i] = MonteCarloSampleInput(global_params,
                                                    mc_params,
                                                    nn_params,
                                                    system_params,
                                                    model_weights)
    end

    # Replicate inputs for all workers
    sets_per_system = n_workers ÷ n_systems
    return repeat(reference_inputs, sets_per_system)
end

function train!(global_params::GlobalParameters,
                mc_params::MonteCarloParameters,
                nn_params::NeuralNetParameters,
                system_params_list::Vector{SystemParameters},
                model::Flux.Chain,
                optimizer,
                opt_state_loaded,
                ref_rdfs)
    opt_state = Flux.setup(optimizer, model)
    if !isnothing(opt_state_loaded)
        opt_state = opt_state_loaded
    end

    od = global_params.output_dir
    lr_config = nn_params.lr_scheduler_config
    lr_state = LRSchedulerState(nn_params.optimizer_config.learning_rate, Float32(Inf), 0, 0)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    training_log_file = joinpath(od, "training_$(timestamp)_summary.csv")
    training_log_io = open(training_log_file, "w")
    println(training_log_io, "# epoch,mae,grad_norm,lr")

    try
        for iteration in 1:(nn_params.iterations)
            iter_string = lpad(iteration, 2, "0")

            println("\n--------------------------------- Iteration $iteration ---------------------------------\n")

            if iteration <= lr_config.warmup_epochs
                warmup_lr = lr_for_epoch(lr_config, nn_params.optimizer_config.learning_rate, iteration)
                if warmup_lr != lr_state.current_lr
                    lr_state.current_lr = warmup_lr
                    Flux.adjust!(opt_state, warmup_lr)
                end
            end
            lr = lr_state.current_lr

            inputs = prepare_monte_carlo_inputs(global_params, mc_params, nn_params, system_params_list, model)
            outputs = pmap(mcsample!, inputs)

            system_outputs,
            system_losses = collect_system_averages(outputs, ref_rdfs, system_params_list, global_params,
                                                    nn_params, model, lr,
                                                    iteration,
                                                    mc_params.steps)

            loss_gradients = Vector{Any}(undef, length(system_outputs))
            for (system_id, system_output) in enumerate(system_outputs)
                system_params = system_params_list[system_id]

                loss_gradients[system_id] = compute_loss_gradients(system_output.cross_accumulators,
                                                                   system_output.symmetry_matrix_accumulator,
                                                                   system_output.descriptor,
                                                                   ref_rdfs[system_id],
                                                                   model,
                                                                   system_params,
                                                                   nn_params)

                name = system_params.system_name
                write_rdf(joinpath(od, "RDFNN-$(name)-iter-$(iter_string).dat"), system_output.descriptor,
                          system_params)
                write_energies(joinpath(od, "energies-$(name)-iter-$(iter_string).dat"),
                               system_output.energies,
                               mc_params,
                               system_params,
                               1)
            end

            mean_loss_gradients = if global_params.adaptive_scaling
                gradient_coeffs = compute_adaptive_gradient_coefficients(system_losses)

                println("\nGradient scaling:")
                for (coeff, params) in zip(gradient_coeffs, system_params_list)
                    println("   System $(params.system_name): $(round(coeff; digits=8))")
                end

                sum(loss_gradients .* gradient_coeffs)
            else
                mean(loss_gradients)
            end

            @save joinpath(od, "model-iter-$(iter_string).bson") model
            @save joinpath(od, "opt-iter-$(iter_string).bson") opt_state
            @save joinpath(od, "gradients-iter-$(iter_string).bson") mean_loss_gradients

            tmp_symm_func_matrix::Matrix{Float32} = zeros(Float32, length(nn_params.g2_functions),
                                                          1)
            tmp_energy_gradients = compute_energy_gradients(tmp_symm_func_matrix, model)
            _, gradient_restructure = Flux.destructure(tmp_energy_gradients)
            grad_norm = norm(mean_loss_gradients)
            mean_loss_gradients = gradient_restructure(mean_loss_gradients)
            update_model!(model, opt_state, mean_loss_gradients)

            avg_mae = sum(system_losses) / Float32(length(system_losses))
            step_plateau!(lr_config, lr_state, opt_state, avg_mae)

            println(@sprintf("Epoch: %d | Steps: %d | MAE: %.3e | |∇|: %.3e | LR: %.2e",
                             iteration, mc_params.steps, avg_mae, grad_norm, lr))

            println(training_log_io,
                    @sprintf("%d,%.17e,%.17e,%.17e", iteration, avg_mae, grad_norm, lr))
            flush(training_log_io)

            GC.gc()
            @everywhere GC.gc()
        end
    finally
        close(training_log_io)
    end
end
