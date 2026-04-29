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

function read_random_frame(sys_params::SystemParameters, rng::Xoroshiro128Plus)
    traj = read_xtc(sys_params)
    nframes = Int(size(traj)) - 1  # пропускаем первый кадр
    frame_id = rand(rng, 1:nframes)
    frame = read_step(traj, frame_id)
    box = lengths(UnitCell(frame))
    return frame, frame_id, box
end

function compute_initial_energies(ref_data::ReferenceData, frame_id::Int, model::Chain)
    hist = ref_data.histograms[frame_id]
    pmf = ref_data.pmf
    symm = combine_matrices_from_reference(ref_data, frame_id)
    e_nn_vector = init_system_energies_vector(symm, model)
    e_nn = sum(e_nn_vector)
    e_pmf = sum(hist .* pmf)
    return symm, hist, e_nn, e_pmf, e_nn_vector
end

function compute_pmf(rdf::AbstractVector{T}, system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_bins = system_params.n_bins
    β = system_params.beta
    pmf = Vector{T}(undef, n_bins)
    repulsion_mask = rdf .== zero(T)
    n_repulsion = count(repulsion_mask)
    first_valid = n_repulsion + 1
    max_pmf = -log(rdf[first_valid]) / β
    next_pmf = -log(rdf[first_valid + 1]) / β
    pmf_gradient = max_pmf - next_pmf
    @inbounds for i in eachindex(pmf)
        pmf[i] = repulsion_mask[i] ? max_pmf + pmf_gradient * (first_valid - i) : -log(rdf[i]) / β
    end
    return pmf ./ 2.0
end

function precompute_reference_data(input::PreComputedInput)::ReferenceData
    nn_params = input.nn_params
    sys_params = input.system_params
    ref_rdf = input.reference_rdf

    pmf = compute_pmf(ref_rdf, sys_params)
    traj = read_xtc(sys_params)
    n_frames = Int(size(traj)) - 1  # пропускаем первый кадр

    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    histograms = Vector{Vector{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    g3_matrices = isempty(nn_params.g3_functions) ? Matrix{Float64}[] :
                  Vector{Matrix{Float64}}(undef, n_frames)
    g9_matrices = isempty(nn_params.g9_functions) ? Matrix{Float64}[] :
                  Vector{Matrix{Float64}}(undef, n_frames)

    for frame_id in 1:n_frames
        frame = read_step(traj, frame_id)
        box = lengths(UnitCell(frame))
        coords = positions(frame)

        distance_matrices[frame_id] = build_distance_matrix(frame)
        histograms[frame_id] = zeros(Float64, sys_params.n_bins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], sys_params)

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

function combine_matrices_from_reference(ref_data::ReferenceData, frame_id::Int)
    if isempty(ref_data.g3_matrices) && isempty(ref_data.g9_matrices)
        return ref_data.g2_matrices[frame_id]
    end
    mats = [ref_data.g2_matrices[frame_id]]
    if !isempty(ref_data.g3_matrices)
        push!(mats, ref_data.g3_matrices[frame_id])
    end
    if !isempty(ref_data.g9_matrices)
        push!(mats, ref_data.g9_matrices[frame_id])
    end
    return hcat(mats...)
end

function compute_gradient!(e_nn::T, e_pmf::T, Δe_nn::T, Δe_pmf::T,
                           symm1::AbstractMatrix{T}, symm2::AbstractMatrix{T},
                           model::Chain, pretrain_params::PreTrainingParameters,
                           nn_params::NeuralNetParameters, use_diff_gradient::Bool)::Any where {T <: AbstractFloat}
    model_params = Flux.trainables(model)
    mp, _ = Flux.destructure(model_params)
    reg_gradient = @. mp * 2 * pretrain_params.regularization

    if !use_diff_gradient
        grad_final = compute_energy_gradients(symm2, model)
        gradient_scale = 2 * (e_nn - e_pmf)
        # gradient_scale = sign(e_nn - e_pmf)
        flat_grad, restructure = Flux.destructure(grad_final)
        loss_gradient = @. gradient_scale * flat_grad
        return restructure(loss_gradient + reg_gradient)
    else
        grad1 = compute_energy_gradients(symm1, model)
        grad2 = compute_energy_gradients(symm2, model)
        gradient_scale = 2 * (Δe_nn - Δe_pmf)
        # gradient_scale = sign(Δe_nn - Δe_pmf)
        flat_grad1, restructure = Flux.destructure(grad1)
        flat_grad2, _ = Flux.destructure(grad2)
        loss_gradient = @. gradient_scale * (flat_grad2 - flat_grad1)
        return restructure(loss_gradient + reg_gradient)
    end
end

function pretraining_move!(ref_data::ReferenceData, model::Flux.Chain,
                           nn_params::NeuralNetParameters, sys_params::SystemParameters,
                           rng::Xoroshiro128Plus)
    frame, frame_id, box = read_random_frame(sys_params, rng)
    point_index = rand(rng, 1:(sys_params.n_atoms))

    # Исходные данные
    distance_matrix = ref_data.distance_matrices[frame_id]
    symm1, hist, e_nn1, e_pmf1, e_nn1_vector = compute_initial_energies(ref_data, frame_id, model)
    distance_vec1 = distance_matrix[:, point_index]

    # Смещение частицы
    dr = sys_params.max_displacement * (rand(rng, Float64, 3) .- 0.5)
    positions(frame)[:, point_index] .+= dr

    point = positions(frame)[:, point_index]
    distance_vec2 = compute_distance_vector(point, positions(frame), box)
    update_distance_histogram_vectors!(hist, distance_vec1, distance_vec2, sys_params)
    update_mask = get_energies_update_mask(distance_vec2, nn_params)

    coords_for_update = (isempty(ref_data.g3_matrices) && isempty(ref_data.g9_matrices)) ? nothing :
                        copy(positions(frame))

    g2_matrix2 = copy(ref_data.g2_matrices[frame_id])
    update_g2_matrix!(g2_matrix2, distance_vec1, distance_vec2, sys_params, nn_params, point_index)

    g3_matrix2 = if !isempty(ref_data.g3_matrices)
        g3_copy = copy(ref_data.g3_matrices[frame_id])
        update_g3_matrix!(g3_copy, coords_for_update, positions(frame), box,
                          distance_vec1, distance_vec2, sys_params, nn_params, point_index)
        g3_copy
    else
        []
    end

    g9_matrix2 = if !isempty(ref_data.g9_matrices)
        g9_copy = copy(ref_data.g9_matrices[frame_id])
        update_g9_matrix!(g9_copy, coords_for_update, positions(frame), box,
                          distance_vec1, distance_vec2, sys_params, nn_params, point_index)
        g9_copy
    else
        []
    end

    symm2 = combine_symmetry_matrices(g2_matrix2, g3_matrix2, g9_matrix2)
    e_nn2_vector = update_system_energies_vector(symm2, model, update_mask, e_nn1_vector)
    e_nn2 = sum(e_nn2_vector)
    e_pmf2 = sum(hist .* ref_data.pmf)

    # Восстанавливаем исходное состояние
    positions(frame)[:, point_index] .-= dr
    update_distance_histogram_vectors!(hist, distance_vec2, distance_vec1, sys_params)

    return (symm1=symm1,
            symm2=symm2,
            Δe_nn=e_nn2 - e_nn1,
            Δe_pmf=e_pmf2 - e_pmf1,
            e_nn1=e_nn1,
            e_pmf1=e_pmf1,
            e_nn2=e_nn2,
            e_pmf2=e_pmf2)
end

function all_particle_move!(ref_data::ReferenceData, model::Flux.Chain,
                            nn_params::NeuralNetParameters, sys_params::SystemParameters,
                            rng::Xoroshiro128Plus)
    frame, frame_id, box = read_random_frame(sys_params, rng)
    symm1, hist, e_nn1, e_pmf1, _ = compute_initial_energies(ref_data, frame_id, model)

    old_coords = copy(positions(frame))
    dr = sys_params.max_displacement * (rand(rng, Float64, 3, sys_params.n_atoms) .- 0.5)
    positions(frame) .+= dr

    distance_matrix2 = build_distance_matrix(frame)
    hist2 = zeros(Float64, sys_params.n_bins)
    update_distance_histogram!(distance_matrix2, hist2, sys_params)

    g2_matrix2 = build_g2_matrix(distance_matrix2, nn_params)
    g3_matrix2 = isempty(nn_params.g3_functions) ? Matrix{Float64}[] :
                 build_g3_matrix(distance_matrix2, positions(frame), box, nn_params)
    g9_matrix2 = isempty(nn_params.g9_functions) ? Matrix{Float64}[] :
                 build_g9_matrix(distance_matrix2, positions(frame), box, nn_params)

    symm2 = combine_symmetry_matrices(g2_matrix2, g3_matrix2, g9_matrix2)
    e_nn2_vector = init_system_energies_vector(symm2, model)
    e_nn2 = sum(e_nn2_vector)
    e_pmf2 = sum(hist2 .* ref_data.pmf)

    positions(frame) .= old_coords

    return (symm1=symm1,
            symm2=symm2,
            Δe_nn=e_nn2 - e_nn1,
            Δe_pmf=e_pmf2 - e_pmf1,
            e_nn1=e_nn1,
            e_pmf1=e_pmf1,
            e_nn2=e_nn2,
            e_pmf2=e_pmf2)
end

function make_mc_move!(use_all_particles::Bool,
                       ref_data::ReferenceData,
                       model::Chain,
                       nn_params::NeuralNetParameters,
                       sys_params::SystemParameters,
                       rng::Xoroshiro128Plus)
    return use_all_particles ?
           all_particle_move!(ref_data, model, nn_params, sys_params, rng) :
           pretraining_move!(ref_data, model, nn_params, sys_params, rng)
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

function run_training_phase!(steps::Int, batch_size::Int,
                             use_diff_gradient::Bool, use_all_particles::Bool,
                             system_params_list, ref_data_list,
                             model::Chain, nn_params::NeuralNetParameters,
                             pretrain_params::PreTrainingParameters,
                             optimizer, lr_schedule::Dict{Int, Float64},
                             initial_lr::Float64; log_prefix::String="pretraining")
    rng = Xoroshiro128Plus()
    n_systems = length(system_params_list)
    opt_state = Flux.setup(optimizer, model)
    current_lr = initial_lr
    Flux.adjust!(opt_state, current_lr)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    phase_type = use_diff_gradient ? "diff" : "abs"
    move_type = use_all_particles ? "all" : "single"
    log_file = "$(log_prefix)_$(phase_type)_$(move_type)_$(timestamp)_detail.dat"
    avg_log_file = "$(log_prefix)_$(phase_type)_$(move_type)_$(timestamp)_summary.dat"

    for epoch in 1:steps
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
                move = make_mc_move!(use_all_particles,
                                     ref_data_list[sys_id],
                                     model,
                                     nn_params,
                                     system_params_list[sys_id],
                                     rng)
                symm1, symm2 = move.symm1, move.symm2
                Δe_nn, Δe_pmf = move.Δe_nn, move.Δe_pmf
                e_nn1, e_pmf1 = move.e_nn1, move.e_pmf1
                e_nn2, e_pmf2 = move.e_nn2, move.e_pmf2

                batch_gradients[sys_id] = compute_gradient!(e_nn2, e_pmf2, Δe_nn, Δe_pmf,
                                                            symm1, symm2, model,
                                                            pretrain_params, nn_params,
                                                            use_diff_gradient)

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
            batch_flat_grad = isnothing(batch_flat_grad) ? first_flat_grad : batch_flat_grad .+ first_flat_grad
        end

        batch_flat_grad ./= batch_size
        final_grad = grad_restructure(batch_flat_grad)
        update_model!(model, opt_state, final_grad)

        if haskey(lr_schedule, epoch)
            current_lr = lr_schedule[epoch]
            Flux.adjust!(opt_state, current_lr)
        end

        mean_mse_diff = accum_mse_diff / count
        mean_mae_diff = accum_mae_diff / count
        mean_mse_abs = accum_mse_abs / count
        mean_mae_abs = accum_mae_abs / count
        mean_reg = accum_reg / count

        log_average_metrics(avg_log_file, epoch,
                            mean_mae_diff, mean_mse_diff,
                            mean_mae_abs, mean_mse_abs,
                            mean_reg)

        println(@sprintf("%s | %s | Epoch: %4d | Batch Size: %3d | Diff MAE: %8.2f | Abs MAE: %8.2f | LR: %.2e",
                         phase_type, move_type, epoch, batch_size, mean_mae_diff, mean_mae_abs, current_lr))
    end

    return model, opt_state
end

function pretrain_model!(pretrain_params::PreTrainingParameters,
                         nn_params::NeuralNetParameters,
                         system_params_list,
                         model::Chain,
                         optimizer,
                         reference_rdfs)
    ref_inputs = [PreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                  for i in 1:length(system_params_list)]
    ref_data_list = pmap(precompute_reference_data, ref_inputs)

    lr_schedule = Dict(5000 => 0.001,
                       30000 => 0.0005,
                       70000 => 0.0001,
                       95000 => 0.00005,
                       99000 => 0.00001)

    # lr_schedule = Dict(10 => 0.0001,
    #                    4900 => 0.00005)

    # @load "pt-model-1.bson" model

    model, opt_state = run_training_phase!(100000,      # Steps
                                           1,           # Batch Size
                                           false,       # Use gradient for Difference
                                           true,        # Move all particles
                                           system_params_list,
                                           ref_data_list,
                                           model,
                                           nn_params,
                                           pretrain_params,
                                           optimizer,
                                           lr_schedule,
                                           pretrain_params.learning_rate,
                                           log_prefix="phase1")

    @save "pt-model-1.bson" model
    @save "pt-opt-state-1.bson" optimizer
    return model
end
