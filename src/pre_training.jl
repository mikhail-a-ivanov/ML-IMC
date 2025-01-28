using ..ML_IMC

struct PreComputedInput
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    reference_rdf::Vector{Float64}
end

struct ReferenceData
    distance_matrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    pmf::Vector{Float64}
    g2_matrices::Vector{Matrix{Float64}}
    g3_matrices::Vector{Matrix{Float64}}
    g9_matrices::Vector{Matrix{Float64}}
end

function compute_pmf(rdf::AbstractVector{T},
                     system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_bins = system_params.n_bins
    β = system_params.beta

    # Initialize PMF array and identify repulsive regions
    pmf = Vector{T}(undef, n_bins)
    repulsion_mask = rdf .== zero(T)
    n_repulsion_points = count(repulsion_mask)

    # Calculate reference PMF values
    first_valid_index = n_repulsion_points + 1
    max_pmf = -log(rdf[first_valid_index]) / β
    next_pmf = -log(rdf[first_valid_index + 1]) / β
    pmf_gradient = max_pmf - next_pmf

    # Fill PMF array
    @inbounds for i in eachindex(pmf)
        if repulsion_mask[i]
            # Extrapolate PMF in repulsive region
            pmf[i] = max_pmf + pmf_gradient * (first_valid_index - i)
        else
            # Calculate PMF from RDF
            pmf[i] = -log(rdf[i]) / β
        end
    end

    return pmf
end

function precompute_reference_data(input::PreComputedInput;)::ReferenceData
    # Unpack input parameters
    nn_params = input.nn_params
    system_params = input.system_params
    ref_rdf = input.reference_rdf

    # Compute potential of mean force
    pmf = compute_pmf(ref_rdf, system_params)

    # Initialize trajectory reading
    trajectory = read_xtc(system_params)
    n_frames = Int(size(trajectory)) - 1  # Skip first frame

    # Pre-allocate data containers
    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    histograms = Vector{Vector{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    g3_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.g3_functions) > 0))
    g9_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.g9_functions) > 0))

    # Process each frame
    for frame_id in 1:n_frames
        frame = read_step(trajectory, frame_id)
        box = lengths(UnitCell(frame))
        coords = positions(frame)

        # Compute distance matrix and histogram
        distance_matrices[frame_id] = build_distance_matrix(frame)
        histograms[frame_id] = zeros(Float64, system_params.n_bins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], system_params)

        # Compute symmetry function matrices
        g2_matrices[frame_id] = build_g2_matrix(distance_matrices[frame_id], nn_params)

        if !isempty(nn_params.g3_functions)
            g3_matrices[frame_id] = build_g3_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end

        if !isempty(nn_params.g9_functions)
            g9_matrices[frame_id] = build_g9_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end
    end

    return ReferenceData(distance_matrices,
                         histograms,
                         pmf,
                         g2_matrices,
                         g3_matrices,
                         g9_matrices)
end

function compute_pretraining_gradient_diff(energy_diff_nn::T,
                                           energy_diff_pmf::T,
                                           symm_func_matrix1::AbstractMatrix{T},
                                           symm_func_matrix2::AbstractMatrix{T},
                                           model::Chain,
                                           pretrain_params::PreTrainingParameters,
                                           nn_params::NeuralNetParameters;) where {T <: AbstractFloat}

    # Calculate regularization loss
    model_params = Flux.trainables(model)

    # Compute gradients for both configurations
    grad1 = compute_energy_gradients(symm_func_matrix1, model)
    grad2 = compute_energy_gradients(symm_func_matrix2, model)

    # Calculate loss gradients with regularization
    loss_type = "mse"

    gradient_scale = if loss_type == "mse"
        2 * (energy_diff_nn - energy_diff_pmf)
    elseif loss_type == "mae"
        sign(energy_diff_nn - energy_diff_pmf)
    else
        throw(ArgumentError("Unsupported loss type: $loss_type. Supported types are: 'mse', 'mae'"))
    end

    flat_grad_1, re_1 = Flux.destructure(grad1)
    flat_grad_2, re_2 = Flux.destructure(grad2)

    mp, re_mp = Flux.destructure(model_params)
    loss_gradient = @. gradient_scale * (flat_grad_2 - flat_grad_1)
    reg_gradient = @. mp * 2 * pretrain_params.regularization

    pretraining_gradient = loss_gradient + reg_gradient
    pretraining_gradient = re_1(pretraining_gradient)

    return pretraining_gradient
end

function compute_pretraining_gradient_abs(e_nn::T,
                                          e_pmf::T,
                                          symm_func_matrix::AbstractMatrix{T},
                                          model::Flux.Chain,
                                          pretrain_params::PreTrainingParameters,
                                          nn_params::NeuralNetParameters)::Any where {T <: AbstractFloat}
    model_params = Flux.trainables(model)
    grad_final = compute_energy_gradients(symm_func_matrix, model)
    diff = e_nn - e_pmf

    # Scaling coefficient depends on loss function type
    loss_type = "mse"
    gradient_scale = if loss_type == "mse"
        2 * diff
    elseif loss_type == "mae"
        sign(diff)
    else
        throw(ArgumentError("Unsupported loss type: $loss_type. Supported: mse, mae"))
    end

    flat_grad_final, re_final = Flux.destructure(grad_final)
    mp, re_mp = Flux.destructure(model_params)
    loss_gradient = @. gradient_scale * flat_grad_final
    reg_gradient = @. mp * 2 * pretrain_params.regularization
    pretraining_gradient = loss_gradient + reg_gradient
    pretraining_gradient = re_final(pretraining_gradient)

    return pretraining_gradient
end

function pretraining_move!(reference_data::ReferenceData,
                           model::Flux.Chain,
                           nn_params::NeuralNetParameters,
                           system_params::SystemParameters,
                           rng::Xoroshiro128Plus)
    traj = read_xtc(system_params)
    nframes = Int(size(traj)) - 1  # skip first frame
    frame_id = rand(rng, 1:nframes)
    frame = read_step(traj, frame_id)
    box = lengths(UnitCell(frame))
    point_index = rand(rng, 1:(system_params.n_atoms))

    distance_matrix = reference_data.distance_matrices[frame_id]
    hist = reference_data.histograms[frame_id]
    pmf = reference_data.pmf

    if isempty(reference_data.g3_matrices) && isempty(reference_data.g9_matrices)
        symm_func_matrix1 = reference_data.g2_matrices[frame_id]
    else
        # Store initial coordinates for G3/G9 matrix updates
        coordinates1 = copy(positions(frame))
        if isempty(reference_data.g3_matrices)
            symm_func_matrices = [
                reference_data.g2_matrices[frame_id],
                reference_data.g9_matrices[frame_id]
            ]
        elseif isempty(reference_data.g9_matrices)
            symm_func_matrices = [
                reference_data.g2_matrices[frame_id],
                reference_data.g3_matrices[frame_id]
            ]
        else
            symm_func_matrices = [
                reference_data.g2_matrices[frame_id],
                reference_data.g3_matrices[frame_id],
                reference_data.g9_matrices[frame_id]
            ]
        end
        symm_func_matrix1 = hcat(symm_func_matrices...)
    end

    e_nn1_vector = init_system_energies_vector(symm_func_matrix1, model)
    e_nn1 = sum(e_nn1_vector)
    e_pmf1 = sum(hist .* pmf)

    distance_vector1 = distance_matrix[:, point_index]
    dr = system_params.max_displacement * (rand(rng, Float64, 3) .- 0.5)
    positions(frame)[:, point_index] .+= dr

    point = positions(frame)[:, point_index]
    distance_vector2 = compute_distance_vector(point, positions(frame), box)
    hist = update_distance_histogram_vectors!(hist, distance_vector1, distance_vector2, system_params)
    indexes_for_update = get_energies_update_mask(distance_vector2, nn_params)

    g2_matrix2 = copy(reference_data.g2_matrices[frame_id])
    update_g2_matrix!(g2_matrix2, distance_vector1, distance_vector2, system_params, nn_params, point_index)

    g3_matrix2 = []
    if !isempty(reference_data.g3_matrices)
        g3_matrix2 = copy(reference_data.g3_matrices[frame_id])
        update_g3_matrix!(g3_matrix2, coordinates1, positions(frame), box,
                          distance_vector1, distance_vector2,
                          system_params, nn_params, point_index)
    end

    g9_matrix2 = []
    if !isempty(reference_data.g9_matrices)
        g9_matrix2 = copy(reference_data.g9_matrices[frame_id])
        update_g9_matrix!(g9_matrix2, coordinates1, positions(frame), box,
                          distance_vector1, distance_vector2,
                          system_params, nn_params, point_index)
    end

    symm_func_matrix2 = combine_symmetry_matrices(g2_matrix2, g3_matrix2, g9_matrix2)

    e_nn2_vector = update_system_energies_vector(symm_func_matrix2, model,
                                                 indexes_for_update,
                                                 e_nn1_vector)
    e_nn2 = sum(e_nn2_vector)
    e_pmf2 = sum(hist .* pmf)

    Δe_nn = e_nn2 - e_nn1
    Δe_pmf = e_pmf2 - e_pmf1

    # Restore initial state
    positions(frame)[:, point_index] .-= dr
    hist = update_distance_histogram_vectors!(hist, distance_vector2, distance_vector1, system_params)

    return (symm_func_matrix1,
            symm_func_matrix2,
            Δe_nn,
            Δe_pmf,
            e_nn1,
            e_pmf1,
            e_nn2,
            e_pmf2)
end

function pretrain_model!(pretrain_params::PreTrainingParameters,
                         nn_params::NeuralNetParameters,
                         system_params_list,
                         model::Chain,
                         optimizer,
                         reference_rdfs;
                         save_path::String="model-pre-trained.bson",
                         verbose::Bool=true,
                         phase1_steps::Int=500,  # Steps for absolute value training
                         phase2_steps::Int=500)  # Steps for energy difference training
    # Log files
    ALL_LOSS_LOG_FILE = "pretraining_loss.out"
    AVG_LOSS_LOG_FILE = "avg_pretraining_loss.out"

    rng = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    n_systems = length(system_params_list)

    # Prepare reference data
    ref_data_inputs = [PreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                       for i in 1:n_systems]
    ref_data_list = pmap(precompute_reference_data, ref_data_inputs)

    # Initialize optimizer state
    opt_state = Flux.setup(optimizer, model)
    lr = optimizer.eta
    batch_size = 100
    steps_per_phase_1 = pretrain_params.steps // 2
    steps_per_phase_2 = pretrain_params.steps // 2
    # steps_per_phase_1 = 50
    # steps_per_phase_2 = 50

    # Learning rate schedules for Phase 1 (Absolute Values)
    phase1_lr_schedule = Dict(500 => 0.005,
                              1000 => 0.001,
                              2500 => 0.0005)
                            #   1000 => 0.0001,
                            #   2000 => 0.00005,
                            #   4500 => 0.00001)

    #=========================
    Phase 1: Training on Absolute Values
    =========================#
    for epoch in 1:steps_per_phase_1
        should_report = (epoch % pretrain_params.output_frequency == 0) || (epoch == 1)

        # Metric accumulators
        accum_mse_diff = 0.0
        accum_mae_diff = 0.0
        accum_mse_abs = 0.0
        accum_mae_abs = 0.0
        accum_reg = 0.0
        accum_count = 0

        pt_accum_flat_grad = nothing
        grad_restructure = nothing

        # Batch loop
        for batch_iter in 1:batch_size
            loss_gradients = Vector{Any}(undef, n_systems)

            for sys_id in 1:n_systems
                ref_data = ref_data_list[sys_id]

                # Get training data
                symm1, symm2, Δe_nn, Δe_pmf,
                e_nn1, e_pmf1,
                e_nn2, e_pmf2 = pretraining_move!(ref_data,
                                                  model,
                                                  nn_params,
                                                  system_params_list[sys_id],
                                                  rng)

                # Phase 1: Compute gradients using absolute values
                loss_gradients[sys_id] = compute_pretraining_gradient_abs(e_nn2,
                                                                          e_pmf2,
                                                                          symm2,
                                                                          model,
                                                                          pretrain_params,
                                                                          nn_params)

                # Compute metrics
                diff_mse = (Δe_nn - Δe_pmf)^2
                diff_mae = abs(Δe_nn - Δe_pmf)
                abs_mse = (e_nn2 - e_pmf2)^2
                abs_mae = abs(e_nn2 - e_pmf2)

                # Compute regularization loss
                model_params = Flux.trainables(model)
                reg_loss = pretrain_params.regularization * sum(x -> sum(abs2, x), model_params)

                # Log individual batch results
                try
                    open(ALL_LOSS_LOG_FILE, "a") do io
                        println(io,
                                @sprintf("%d %d %d  %.8f %.8f  %.8f %.8f  %.8f %.8f  %.8f %.8f",
                                         epoch,
                                         batch_iter,
                                         sys_id,
                                         diff_mae,
                                         diff_mse,
                                         abs_mae,
                                         abs_mse,
                                         e_nn2,
                                         e_pmf2,
                                         Δe_nn,
                                         Δe_pmf))
                    end
                catch e
                    @warn "Failed to write to all-iteration log file: $ALL_LOSS_LOG_FILE" exception=e
                end

                # Accumulate metrics
                accum_mse_diff += diff_mse
                accum_mae_diff += diff_mae
                accum_mse_abs += abs_mse
                accum_mae_abs += abs_mae
                accum_reg += reg_loss
                accum_count += 1
            end

            # Combine gradients from all systems
            first_flat_grad, grad_restructure = Flux.destructure(loss_gradients[1])
            for grad in loss_gradients[2:end]
                flat_grad, _ = Flux.destructure(grad)
                first_flat_grad .+= flat_grad
            end
            first_flat_grad ./= n_systems

            # Accumulate gradients within batch
            if pt_accum_flat_grad === nothing
                pt_accum_flat_grad = first_flat_grad
            else
                pt_accum_flat_grad .+= first_flat_grad
            end
        end

        # Average gradients and update model
        pt_accum_flat_grad ./= batch_size
        final_struct_grad = grad_restructure(pt_accum_flat_grad)
        update_model!(model, opt_state, final_struct_grad)

        # Save checkpoint and adjust learning rate
        if epoch % 5 == 0 || epoch == 1
            checkpoint_path = "pt_checkpoint_phase1_$(epoch).bson"
            # @save checkpoint_path model

            # Check for learning rate adjustment
            if haskey(phase1_lr_schedule, epoch)
                lr = phase1_lr_schedule[epoch]
                Flux.adjust!(opt_state, lr)
            end
        end

        # Log average metrics
        mean_mse_diff = accum_mse_diff / accum_count
        mean_mae_diff = accum_mae_diff / accum_count
        mean_mse_abs = accum_mse_abs / accum_count
        mean_mae_abs = accum_mae_abs / accum_count
        mean_reg = accum_reg / accum_count

        try
            open(AVG_LOSS_LOG_FILE, "a") do io
                println(io,
                        @sprintf("%d %.8f %.8f %.8f %.8f %.2e",
                                 epoch,
                                 mean_mae_diff,
                                 mean_mse_diff,
                                 mean_mae_abs,
                                 mean_mse_abs,
                                 mean_reg))
            end
        catch e
            @warn "Failed to write to avg-log file: $AVG_LOSS_LOG_FILE" exception=e
        end

        if should_report
            println(@sprintf("Abs - Epoch: %4d | diff_MSE: %.6f | diff_MAE: %.6f | abs_MSE: %.6f | abs_MAE: %.6f | Reg: %.2e | lr: %.2e",
                             epoch,
                             mean_mse_diff,
                             mean_mae_diff,
                             mean_mse_abs,
                             mean_mae_abs,
                             mean_reg,
                             lr))
        end
    end

    # Save model after Phase 1
    @save "model-phase1.bson" model

    # Learning rate schedules for Phase 2 (Energy Differences)
    Flux.adjust!(opt_state, 0.001)
    phase2_lr_schedule = Dict(500 => 0.0005,
                              1000 => 0.0001,
                              2000 => 0.00005,
                              4500 => 0.00001)

    #=========================
    Phase 2: Training on Energy Differences
    =========================#
    for epoch in 1:steps_per_phase_2
        should_report = (epoch % pretrain_params.output_frequency == 0) || (epoch == 1)

        # Metric accumulators
        accum_mse_diff = 0.0
        accum_mae_diff = 0.0
        accum_mse_abs = 0.0
        accum_mae_abs = 0.0
        accum_reg = 0.0
        accum_count = 0

        pt_accum_flat_grad = nothing
        grad_restructure = nothing

        # Batch loop
        for batch_iter in 1:batch_size
            loss_gradients = Vector{Any}(undef, n_systems)

            for sys_id in 1:n_systems
                ref_data = ref_data_list[sys_id]

                # Get training data
                symm1, symm2, Δe_nn, Δe_pmf,
                e_nn1, e_pmf1,
                e_nn2, e_pmf2 = pretraining_move!(ref_data,
                                                  model,
                                                  nn_params,
                                                  system_params_list[sys_id],
                                                  rng)

                # Phase 2: Compute gradients using energy differences
                loss_gradients[sys_id] = compute_pretraining_gradient_diff(Δe_nn,
                                                                           Δe_pmf,
                                                                           symm1,
                                                                           symm2,
                                                                           model,
                                                                           pretrain_params,
                                                                           nn_params)

                # Compute metrics
                diff_mse = (Δe_nn - Δe_pmf)^2
                diff_mae = abs(Δe_nn - Δe_pmf)
                abs_mse = (e_nn2 - e_pmf2)^2
                abs_mae = abs(e_nn2 - e_pmf2)

                # Compute regularization loss
                model_params = Flux.trainables(model)
                reg_loss = pretrain_params.regularization * sum(x -> sum(abs2, x), model_params)

                # Log individual batch results
                try
                    open(ALL_LOSS_LOG_FILE, "a") do io
                        println(io,
                                @sprintf("%d %d %d  %.8f %.8f  %.8f %.8f  %.8f %.8f  %.8f %.8f",
                                         epoch,
                                         batch_iter,
                                         sys_id,
                                         diff_mae,
                                         diff_mse,
                                         abs_mae,
                                         abs_mse,
                                         e_nn2,
                                         e_pmf2,
                                         Δe_nn,
                                         Δe_pmf))
                    end
                catch e
                    @warn "Failed to write to all-iteration log file: $ALL_LOSS_LOG_FILE" exception=e
                end

                # Accumulate metrics
                accum_mse_diff += diff_mse
                accum_mae_diff += diff_mae
                accum_mse_abs += abs_mse
                accum_mae_abs += abs_mae
                accum_reg += reg_loss
                accum_count += 1
            end

            # Combine gradients from all systems
            first_flat_grad, grad_restructure = Flux.destructure(loss_gradients[1])
            for grad in loss_gradients[2:end]
                flat_grad, _ = Flux.destructure(grad)
                first_flat_grad .+= flat_grad
            end
            first_flat_grad ./= n_systems

            # Accumulate gradients within batch
            if pt_accum_flat_grad === nothing
                pt_accum_flat_grad = first_flat_grad
            else
                pt_accum_flat_grad .+= first_flat_grad
            end
        end

        # Average gradients and update model
        pt_accum_flat_grad ./= batch_size
        final_struct_grad = grad_restructure(pt_accum_flat_grad)
        update_model!(model, opt_state, final_struct_grad)

        # Save checkpoint and adjust learning rate
        if epoch % 5 == 0 || epoch == 1
            checkpoint_path = "pt_checkpoint_phase2_$(epoch).bson"
            # @save checkpoint_path model

            # Check for learning rate adjustment
            if haskey(phase2_lr_schedule, epoch)
                lr = phase2_lr_schedule[epoch]
                Flux.adjust!(opt_state, lr)
            end
        end

        # Log average metrics
        mean_mse_diff = accum_mse_diff / accum_count
        mean_mae_diff = accum_mae_diff / accum_count
        mean_mse_abs = accum_mse_abs / accum_count
        mean_mae_abs = accum_mae_abs / accum_count
        mean_reg = accum_reg / accum_count

        try
            open(AVG_LOSS_LOG_FILE, "a") do io
                println(io,
                        @sprintf("%d %.8f %.8f %.8f %.8f %.2e",
                                 epoch,
                                 mean_mae_diff,
                                 mean_mse_diff,
                                 mean_mae_abs,
                                 mean_mse_abs,
                                 mean_reg))
            end
        catch e
            @warn "Failed to write to avg-log file: $AVG_LOSS_LOG_FILE" exception=e
        end

        if should_report
            println(@sprintf("Diff - Epoch: %4d | diff_MSE: %.6f | diff_MAE: %.6f | abs_MSE: %.6f | abs_MAE: %.6f | Reg: %.2e | lr: %.2e",
                             epoch,
                             mean_mse_diff,
                             mean_mae_diff,
                             mean_mse_abs,
                             mean_mae_abs,
                             mean_reg,
                             lr))
        end
    end

    # Save final model
    try
        @save save_path model
        check_file(save_path)
    catch e
        @warn "Failed to save model to $save_path" exception=e
    end

    return model
end
