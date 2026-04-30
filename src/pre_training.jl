using ..ML_IMC

struct PreComputedInput
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    reference_rdf::Vector{Float64}
end

struct ReferenceData
    cache::PretrainingReferenceCache
    pmf::Vector{Float64}
    pmf_energies::Vector{Float64}
end

function compute_initial_energies(ref_data::ReferenceData, frame_id::Int, model::Chain)
    symm = ref_data.cache.g2_matrices[frame_id]
    e_nn_vector = init_system_energies_vector(symm, model)
    e_nn = sum(e_nn_vector)
    e_pmf = ref_data.pmf_energies[frame_id]
    return symm, e_nn, e_pmf, e_nn_vector
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

function compute_histogram_energy(histogram::AbstractVector{T},
                                  pair_energies::AbstractVector{T})::T where {T <: AbstractFloat}
    energy = zero(T)

    @inbounds @simd for i in eachindex(histogram, pair_energies)
        energy += histogram[i] * pair_energies[i]
    end

    return energy
end

function compute_binned_pair_energy_delta(old_distances::AbstractVector{T},
                                          new_distances::AbstractVector{T},
                                          pair_energies::AbstractVector{T},
                                          system_params::SystemParameters,
                                          point_index::Integer)::T where {T <: AbstractFloat}
    delta = zero(T)
    n_bins = system_params.n_bins
    bin_width = system_params.bin_width

    @inbounds for i in eachindex(old_distances, new_distances)
        i == point_index && continue

        old_bin = floor(Int, 1 + old_distances[i] / bin_width)
        if 1 <= old_bin <= n_bins
            delta -= pair_energies[old_bin]
        end

        new_bin = floor(Int, 1 + new_distances[i] / bin_width)
        if 1 <= new_bin <= n_bins
            delta += pair_energies[new_bin]
        end
    end

    return delta
end

function precompute_reference_data(input::PreComputedInput)::ReferenceData
    nn_params = input.nn_params
    sys_params = input.system_params
    ref_rdf = input.reference_rdf

    pmf = compute_pmf(ref_rdf, sys_params)
    ref_cache = precompute_reference_cache(nn_params, sys_params)
    pmf_energies = [compute_histogram_energy(histogram, pmf)
                    for histogram in ref_cache.histograms]

    return ReferenceData(ref_cache, pmf, pmf_energies)
end

function pretraining_move!(ref_data::ReferenceData, model::Flux.Chain,
                           nn_params::NeuralNetParameters, sys_params::SystemParameters,
                           rng::Xoroshiro128Plus)
    coordinates, frame_id, box = sample_reference_frame(ref_data.cache, rng)
    point_index = rand(rng, 1:(sys_params.n_atoms))

    distance_matrix = ref_data.cache.distance_matrices[frame_id]
    symm1, e_nn1, e_pmf1, e_nn1_vector = compute_initial_energies(ref_data, frame_id, model)
    distance_vec1 = @view distance_matrix[:, point_index]

    point = random_displaced_position(coordinates, point_index, box, sys_params.max_displacement, rng)
    distance_vec2 = compute_distance_vector(point, coordinates, box)
    distance_vec2[point_index] = 0.0

    update_mask = get_energies_update_mask(distance_vec1, distance_vec2, nn_params)

    g2_matrix2 = copy(ref_data.cache.g2_matrices[frame_id])
    update_g2_matrix!(g2_matrix2, distance_vec1, distance_vec2, sys_params, nn_params, point_index)

    symm2 = g2_matrix2
    e_nn2_vector = update_system_energies_vector(symm2, model, update_mask, e_nn1_vector)
    e_nn2 = sum(e_nn2_vector)
    e_pmf2 = e_pmf1 + compute_binned_pair_energy_delta(distance_vec1, distance_vec2, ref_data.pmf,
                                              sys_params, point_index)

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
    coordinates, frame_id, box = sample_reference_frame(ref_data.cache, rng)
    symm1, e_nn1, e_pmf1, _ = compute_initial_energies(ref_data, frame_id, model)

    coordinates2 = random_displaced_coordinates(coordinates, box, sys_params.max_displacement, rng)
    distance_matrix2 = build_distance_matrix(coordinates2, box)
    hist2 = zeros(Float64, sys_params.n_bins)
    update_distance_histogram!(distance_matrix2, hist2, sys_params)

    g2_matrix2 = build_g2_matrix(distance_matrix2, nn_params)

    symm2 = g2_matrix2
    e_nn2_vector = init_system_energies_vector(symm2, model)
    e_nn2 = sum(e_nn2_vector)
    e_pmf2 = compute_histogram_energy(hist2, ref_data.pmf)

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

function run_training_phase!(steps::Int, batch_size::Int,
                             use_diff_gradient::Bool, use_all_particles::Bool,
                             system_params_list, ref_data_list,
                             model::Chain, nn_params::NeuralNetParameters,
                             pretrain_params::PreTrainingParameters,
                             optimizer; log_prefix::String="pretraining")
    rng = Xoroshiro128Plus()
    n_systems = length(system_params_list)
    opt_state = Flux.setup(optimizer, model)
    lr_config = pretrain_params.lr_scheduler_config
    initial_lr = pretrain_params.learning_rate
    lr_state = LRSchedulerState(initial_lr, Inf, 0, 0)
    Flux.adjust!(opt_state, initial_lr)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    phase_type = use_diff_gradient ? "diff" : "abs"
    move_type = use_all_particles ? "all" : "single"
    od = pretrain_params.output_dir
    summary_file = joinpath(od, "$(log_prefix)_$(phase_type)_$(move_type)_$(timestamp)_summary.csv")

    summary_io = open(summary_file, "w")
    println(summary_io, "# epoch,mean_diff_mae,mean_abs_mae,grad_norm,lr")

    try
        for epoch in 1:steps
            accum_mae_diff = 0.0
            accum_mae_abs = 0.0
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
                    e_nn2, e_pmf2 = move.e_nn2, move.e_pmf2
                    n_atoms = system_params_list[sys_id].n_atoms

                    norm = if use_diff_gradient && !use_all_particles
                        1.0
                    else
                        Float64(n_atoms)
                    end

                    batch_gradients[sys_id] = compute_pretraining_gradient!(e_nn2, e_pmf2, Δe_nn, Δe_pmf,
                                                                            symm1, symm2, model,
                                                                            pretrain_params,
                                                                            use_diff_gradient,
                                                                            norm_factor=norm)

                    accum_mae_diff += abs(Δe_nn - Δe_pmf) / n_atoms
                    accum_mae_abs += abs(e_nn2 - e_pmf2) / n_atoms
                    count += 1
                end

                flat_grad, grad_restructure = mean_gradient(batch_gradients)
                batch_flat_grad = isnothing(batch_flat_grad) ? flat_grad : batch_flat_grad .+ flat_grad
            end

            batch_flat_grad ./= batch_size
            grad_norm = norm(batch_flat_grad)
            final_grad = grad_restructure(batch_flat_grad)
            update_model!(model, opt_state, final_grad)

            if epoch <= lr_config.warmup_epochs
                warmup_lr = lr_for_epoch(lr_config, initial_lr, epoch)
                if warmup_lr != lr_state.current_lr
                    lr_state.current_lr = warmup_lr
                    Flux.adjust!(opt_state, warmup_lr)
                end
            end

            mean_mae_diff = accum_mae_diff / count
            mean_mae_abs = accum_mae_abs / count

            log_pretraining_summary(summary_io, epoch, mean_mae_diff, mean_mae_abs,
                                    grad_norm, lr_state.current_lr)

            plateau_metric = use_diff_gradient ? mean_mae_diff : mean_mae_abs
            step_plateau!(lr_config, lr_state, opt_state, plateau_metric)

            println(@sprintf("PMF PT | %s | %s | Epoch: %4d | Batch: %3d | DiffMAE: %.3e | AbsMAE: %.3e | |∇|: %.3e | LR: %.2e",
                             phase_type, move_type, epoch, batch_size, mean_mae_diff, mean_mae_abs,
                             grad_norm, lr_state.current_lr))

            if epoch % pretrain_params.save_frequency == 0 || epoch == steps
                @save joinpath(od, "$(log_prefix)-model-$(phase_type)-$(move_type)-epoch-$(epoch).bson") model
                @save joinpath(od, "$(log_prefix)-opt-state-$(phase_type)-$(move_type)-epoch-$(epoch).bson") opt_state
            end

            flush(summary_io)
        end
    finally
        close(summary_io)
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

    model,
    opt_state = run_training_phase!(pretrain_params.steps,
                                    pretrain_params.batch_size,
                                    pretrain_params.use_diff_gradient,
                                    pretrain_params.use_all_particles,
                                    system_params_list,
                                    ref_data_list,
                                    model,
                                    nn_params,
                                    pretrain_params,
                                    optimizer,
                                    log_prefix=pretrain_params.output_prefix)

    prefix = pretrain_params.output_prefix
    od = pretrain_params.output_dir
    @save joinpath(od, "$prefix-model-1.bson") model
    @save joinpath(od, "$prefix-opt-state-1.bson") opt_state
    return model
end
