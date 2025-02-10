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

function compute_pmf(rdf::AbstractVector{T}, system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_bins = system_params.n_bins
    β = system_params.beta
    pmf = Vector{T}(undef, n_bins)
    repulsion_mask = rdf .== zero(T)
    n_repulsion_points = count(repulsion_mask)
    first_valid_index = n_repulsion_points + 1
    max_pmf = -log(rdf[first_valid_index]) / β
    next_pmf = -log(rdf[first_valid_index + 1]) / β
    pmf_gradient = max_pmf - next_pmf
    @inbounds for i in eachindex(pmf)
        if repulsion_mask[i]
            pmf[i] = max_pmf + pmf_gradient * (first_valid_index - i)
        else
            pmf[i] = -log(rdf[i]) / β
        end
    end
    return pmf
end

function precompute_reference_data(input::PreComputedInput)::ReferenceData
    nn_params = input.nn_params
    system_params = input.system_params
    ref_rdf = input.reference_rdf

    pmf = compute_pmf(ref_rdf, system_params)
    trajectory = read_xtc(system_params)
    n_frames = Int(size(trajectory)) - 1  # skip first frame

    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    histograms = Vector{Vector{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    g3_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.g3_functions) > 0))
    g9_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.g9_functions) > 0))

    for frame_id in 1:n_frames
        frame = read_step(trajectory, frame_id)
        box = lengths(UnitCell(frame))
        coords = positions(frame)

        distance_matrices[frame_id] = build_distance_matrix(frame)
        histograms[frame_id] = zeros(Float64, system_params.n_bins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], system_params)

        g2_matrices[frame_id] = build_g2_matrix(distance_matrices[frame_id], nn_params)
        if !isempty(nn_params.g3_functions)
            g3_matrices[frame_id] = build_g3_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end
        if !isempty(nn_params.g9_functions)
            g9_matrices[frame_id] = build_g9_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end
    end

    return ReferenceData(distance_matrices, histograms, pmf, g2_matrices, g3_matrices, g9_matrices)
end

function compute_pretraining_gradient_diff(energy_diff_nn::T,
                                           energy_diff_pmf::T,
                                           symm_func_matrix1::AbstractMatrix{T},
                                           symm_func_matrix2::AbstractMatrix{T},
                                           model::Chain,
                                           pretrain_params::PreTrainingParameters,
                                           nn_params::NeuralNetParameters)::Any where {T <: AbstractFloat}
    model_params = Flux.trainables(model)
    grad1 = compute_energy_gradients(symm_func_matrix1, model)
    grad2 = compute_energy_gradients(symm_func_matrix2, model)
    loss_type = "mse"
    gradient_scale = loss_type == "mse" ? 2 * (energy_diff_nn - energy_diff_pmf) :
                     sign(energy_diff_nn - energy_diff_pmf)
    flat_grad_1, re_1 = Flux.destructure(grad1)
    flat_grad_2, _ = Flux.destructure(grad2)
    mp, _ = Flux.destructure(model_params)
    loss_gradient = @. gradient_scale * (flat_grad_2 - flat_grad_1)
    reg_gradient = @. mp * 2 * pretrain_params.regularization
    return re_1(loss_gradient + reg_gradient)
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
    loss_type = "mse"
    gradient_scale = loss_type == "mse" ? 2 * diff : sign(diff)
    flat_grad_final, re_final = Flux.destructure(grad_final)
    mp, _ = Flux.destructure(model_params)
    loss_gradient = @. gradient_scale * flat_grad_final
    reg_gradient = @. mp * 2 * pretrain_params.regularization
    return re_final(loss_gradient + reg_gradient)
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

    positions(frame)[:, point_index] .-= dr  # restore state
    hist = update_distance_histogram_vectors!(hist, distance_vector2, distance_vector1, system_params)

    return (symm_func_matrix1,
            symm_func_matrix2,
            e_nn2 - e_nn1,
            e_pmf2 - e_pmf1,
            e_nn1,
            e_pmf1,
            e_nn2,
            e_pmf2)
end

function log_batch_metrics(file::String, epoch, batch_iter, sys_id,
                           diff_mae, diff_mse, abs_mae, abs_mse,
                           e_nn2, e_pmf2, Δe_nn, Δe_pmf)
    try
        open(file, "a") do io
            println(io,
                    @sprintf("%d %d %d  %.8f %.8f  %.8f %.8f  %.8f %.8f  %.8f %.8f",
                             epoch, batch_iter, sys_id,
                             diff_mae, diff_mse,
                             abs_mae, abs_mse,
                             e_nn2, e_pmf2,
                             Δe_nn, Δe_pmf))
        end
    catch e
        @warn "Failed to write to log file: $file" exception=e
    end
end

function log_average_metrics(file::String, epoch,
                             mean_mae_diff, mean_mse_diff,
                             mean_mae_abs, mean_mse_abs, mean_reg)
    try
        open(file, "a") do io
            println(io,
                    @sprintf("%d %.8f %.8f %.8f %.8f %.2e",
                             epoch,
                             mean_mae_diff, mean_mse_diff,
                             mean_mae_abs, mean_mse_abs,
                             mean_reg))
        end
    catch e
        @warn "Failed to write average metrics to file: $file" exception=e
    end
end

function training_phase!(phase_name::String,
                         steps::Int,
                         batch_size::Int,
                         system_params_list,
                         ref_data_list,
                         model::Chain,
                         nn_params::NeuralNetParameters,
                         pretrain_params::PreTrainingParameters,
                         optimizer,
                         rng::Xoroshiro128Plus,
                         gradient_func::Function,
                         lr_schedule::Dict{Int, Float64},
                         log_file::String,
                         avg_log_file::String)
    n_systems = length(system_params_list)
    opt_state = Flux.setup(optimizer, model)
    lr = pretrain_params.learning_rate
    for epoch in 1:steps
        should_report = (epoch % pretrain_params.output_frequency == 0) || (epoch == 1)
        accum_mse_diff = 0.0
        accum_mae_diff = 0.0
        accum_mse_abs = 0.0
        accum_mae_abs = 0.0
        accum_reg = 0.0
        count = 0
        batch_flat_grad = nothing
        grad_restructure = nothing

        for batch_iter in 1:batch_size
            batch_gradients = Vector{Any}(undef, n_systems)
            for sys_id in 1:n_systems
                ref_data = ref_data_list[sys_id]
                symm1, symm2, Δe_nn, Δe_pmf, e_nn1, e_pmf1, e_nn2, e_pmf2 = pretraining_move!(ref_data, model,
                                                                                              nn_params,
                                                                                              system_params_list[sys_id],
                                                                                              rng)
                if phase_name == "abs"
                    batch_gradients[sys_id] = gradient_func(e_nn2, e_pmf2, symm2, model, pretrain_params, nn_params)
                elseif phase_name == "diff"
                    batch_gradients[sys_id] = gradient_func(Δe_nn, Δe_pmf, symm1, symm2, model, pretrain_params,
                                                            nn_params)
                else
                    error("Unknown phase: $phase_name")
                end
                diff_mse = (Δe_nn - Δe_pmf)^2
                diff_mae = abs(Δe_nn - Δe_pmf)
                abs_mse = (e_nn2 - e_pmf2)^2
                abs_mae = abs(e_nn2 - e_pmf2)
                model_params = Flux.trainables(model)
                reg_loss = pretrain_params.regularization * sum(x -> sum(abs2, x), model_params)
                accum_mse_diff += diff_mse
                accum_mae_diff += diff_mae
                accum_mse_abs += abs_mse
                accum_mae_abs += abs_mae
                accum_reg += reg_loss
                count += 1
                log_batch_metrics(log_file, epoch, batch_iter, sys_id,
                                  diff_mae, diff_mse, abs_mae, abs_mse,
                                  e_nn2, e_pmf2, Δe_nn, Δe_pmf)
            end

            first_flat_grad, grad_restructure = Flux.destructure(batch_gradients[1])
            for grad in batch_gradients[2:end]
                flat_grad, _ = Flux.destructure(grad)
                first_flat_grad .+= flat_grad
            end
            first_flat_grad ./= n_systems
            if batch_flat_grad === nothing
                batch_flat_grad = first_flat_grad
            else
                batch_flat_grad .+= first_flat_grad
            end
        end

        batch_flat_grad ./= batch_size
        final_grad = grad_restructure(batch_flat_grad)
        update_model!(model, opt_state, final_grad)

        if haskey(lr_schedule, epoch)
            lr = lr_schedule[epoch]
            Flux.adjust!(opt_state, lr)
        end

        mean_mse_diff = accum_mse_diff / count
        mean_mae_diff = accum_mae_diff / count
        mean_mse_abs = accum_mse_abs / count
        mean_mae_abs = accum_mae_abs / count
        mean_reg = accum_reg / count

        log_average_metrics(avg_log_file, epoch, mean_mae_diff, mean_mse_diff, mean_mae_abs, mean_mse_abs, mean_reg)

            println(@sprintf("%s - Epoch: %4d | diff_MSE: %.6f | diff_MAE: %.6f | abs_MSE: %.6f | abs_MAE: %.6f | Reg: %.2e | LR: %.2e",
                             phase_name, epoch, mean_mse_diff, mean_mae_diff, mean_mse_abs, mean_mae_abs, mean_reg, lr))
    end
end

function pretrain_model!(pretrain_params::PreTrainingParameters,
                         nn_params::NeuralNetParameters,
                         system_params_list,
                         model::Chain,
                         optimizer,
                         reference_rdfs;
                         save_path::String="model-pre-trained.bson",
                         verbose::Bool=true)
    rng = Xoroshiro128Plus()
    n_systems = length(system_params_list)
    ref_data_inputs = [PreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                       for i in 1:n_systems]
    ref_data_list = pmap(precompute_reference_data, ref_data_inputs)

    phase1_steps = pretrain_params.steps
    do_phase_2 = false
    phase2_steps = 1000
    batch_size = pretrain_params.batch_size

    phase1_lr_schedule = Dict(2000 => 0.001,
                              7000 => 0.0005,
                              18000 => 0.0001)

    phase2_lr_schedule = Dict(500 => 0.0002,
                              1000 => 0.0001,
                              2000 => 0.00005,
                              4500 => 0.00001)

    all_loss_log = "pretraining_loss.out"
    avg_loss_log = "avg_pretraining_loss.out"

    training_phase!("abs", phase1_steps, batch_size,
                    system_params_list, ref_data_list,
                    model, nn_params, pretrain_params,
                    optimizer, rng, compute_pretraining_gradient_abs,
                    phase1_lr_schedule, all_loss_log, avg_loss_log)
    @save "model-phase1-stage1.bson" model

    batch_size = 256
    phase1_steps = 1000
    Flux.adjust!(Flux.setup(optimizer, model), 0.00005)
    println()
    training_phase!("abs", phase1_steps, batch_size,
                    system_params_list, ref_data_list,
                    model, nn_params, pretrain_params,
                    optimizer, rng, compute_pretraining_gradient_abs,
                    phase1_lr_schedule, all_loss_log, avg_loss_log)
    @save "model-phase1-stage2.bson" model

    if do_phase_2
        Flux.adjust!(Flux.setup(optimizer, model), 0.00005)
        training_phase!("diff", phase2_steps, batch_size,
                        system_params_list, ref_data_list,
                        model, nn_params, pretrain_params,
                        optimizer, rng, compute_pretraining_gradient_diff,
                        phase2_lr_schedule, all_loss_log, avg_loss_log)
    end

    try
        @save save_path model
    catch e
        @warn "Failed to save model to $save_path" exception=e
    end

    return model
end
