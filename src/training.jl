using ..ML_IMC

function compute_training_loss(descriptor_nn::AbstractVector{T},
                               descriptor_ref::AbstractVector{T},
                               model::Flux.Chain,
                               nn_params::NeuralNetParameters,
                               output_dir::String) where {T <: AbstractFloat}

    # Compute descriptor difference loss
    descriptor_loss_sse = sum(abs2, descriptor_nn .- descriptor_ref)
    descriptor_loss_mse = mean(abs2, descriptor_nn .- descriptor_ref)
    descriptor_loss_rmse = sqrt(mean(abs2, descriptor_nn .- descriptor_ref))
    descriptor_loss_mae = mean(abs, descriptor_nn .- descriptor_ref)

    # Compute L2 regularization loss if regularization parameter is positive
    reg_loss = zero(T)
    if nn_params.regularization > zero(T)
        reg_loss = nn_params.regularization * sum(sum(abs2, p) for p in Flux.trainables(model))
    end

    total_loss_sse = descriptor_loss_sse + reg_loss
    total_loss_mse = descriptor_loss_mse + reg_loss
    total_loss_rmse = descriptor_loss_rmse + reg_loss
    total_loss_mae = descriptor_loss_mae + reg_loss

    # Log descriptor loss to file
    LOSS_LOG_FILE = joinpath(output_dir, "training-loss-values-sse.out")
    try
        open(LOSS_LOG_FILE, "a") do io
            println(io, round(descriptor_loss_sse; digits=8))
        end
        check_file(LOSS_LOG_FILE)
    catch e
        @warn "Failed to log loss value" exception=e
    end

    LOSS_LOG_FILE = joinpath(output_dir, "training-loss-values-mae.out")
    try
        open(LOSS_LOG_FILE, "a") do io
            println(io, round(descriptor_loss_mae; digits=8))
        end
        check_file(LOSS_LOG_FILE)
    catch e
        @warn "Failed to log loss value" exception=e
    end

    return total_loss_mae, total_loss_sse, total_loss_mse, total_loss_rmse
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

    # Create input for each system
    reference_inputs = Vector{MonteCarloSampleInput}(undef, n_systems)
    for (i, system_params) in enumerate(system_params_list)
        reference_inputs[i] = MonteCarloSampleInput(global_params,
                                                    mc_params,
                                                    nn_params,
                                                    system_params,
                                                    model)
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
    lr_state = LRSchedulerState(nn_params.learning_rate, Inf, 0, 0)

    for iteration in 1:(nn_params.iterations)
        iter_string = lpad(iteration, 2, "0")

        println("\n--------------------------------- Iteration $iteration ---------------------------------\n")

        # Apply warmup LR
        warmup_lr = lr_for_epoch(lr_config, nn_params.learning_rate, iteration)
        if warmup_lr != lr_state.current_lr
            lr_state.current_lr = warmup_lr
            Flux.adjust!(opt_state, warmup_lr)
        end
        lr = lr_state.current_lr

        # Monte Carlo sampling
        inputs = prepare_monte_carlo_inputs(global_params, mc_params, nn_params, system_params_list, model)
        outputs = pmap(mcsample!, inputs)

        # Process system outputs and compute losses
        system_outputs,
        system_losses = collect_system_averages(outputs, ref_rdfs, system_params_list, global_params,
                                                nn_params, model, lr,
                                                iteration,
                                                mc_params.steps)

        # Compute gradients for each system
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

            # Save system outputs
            name = system_params.system_name
            write_rdf(joinpath(od, "RDFNN-$(name)-iter-$(iter_string).dat"), system_output.descriptor, system_params)
            write_energies(joinpath(od, "energies-$(name)-iter-$(iter_string).dat"),
                           system_output.energies,
                           mc_params,
                           system_params,
                           1)
        end

        # Compute mean loss gradients
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

        # Save model state
        @save joinpath(od, "model-iter-$(iter_string).bson") model
        @save joinpath(od, "opt-iter-$(iter_string).bson") opt_state
        @save joinpath(od, "gradients-iter-$(iter_string).bson") mean_loss_gradients

        # Update model with computed gradients
        tmp_symm_func_matrix::Matrix{Float64} = zeros(1,
                                                      length(nn_params.g2_functions))
        tmp_energy_gradients = compute_energy_gradients(tmp_symm_func_matrix, model)
        _, gradient_restructure = Flux.destructure(tmp_energy_gradients)
        mean_loss_gradients = gradient_restructure(mean_loss_gradients)
        update_model!(model, opt_state, mean_loss_gradients)

        # Step LR plateau scheduler
        try
            avg_loss = sum(system_losses) / length(system_losses)
            step_plateau!(lr_config, lr_state, opt_state, avg_loss)
        catch e
            @warn "LR plateau step failed" exception=e
        end

        # Run GC after each iteration
        GC.gc()
    end
end
