using ..ML_IMC
using Flux
using Statistics
using RandomNumbers.Xorshifts
using Printf
using Distributed
using Dates
using BSON: @save, @load

struct MagicPreComputedInput
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    reference_rdf::Vector{Float64}
end

struct MagicReferenceData
    distance_matrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    precomputed_energy::Vector{Float64}
    g2_matrices::Vector{Matrix{Float64}}
end

struct PotentialData
    r_values::Vector{Float64}
    potential_values::Vector{Float64}
end

struct PotentialLookup
    lookup_r::Vector{Float64}
    lookup_v::Vector{Float64}
    r_step::Float64
    r_min::Float64
    r_max::Float64
end

struct CachedTrajectory
    frames::Vector{Frame}
    n_frames::Int
    boxes::Vector{Vector{Float64}}
end

function load_potential_data(filename::AbstractString)::PotentialData
    check_file(filename)

    r_values = Float64[]
    potential_values = Float64[]

    open(filename, "r") do file
        for line in eachline(file)
            stripped_line = strip(line)
            isempty(stripped_line) && continue
            startswith(stripped_line, "#") && continue

            values = split(stripped_line)
            length(values) < 2 && continue

            push!(r_values, parse(Float64, values[1]))
            push!(potential_values, parse(Float64, values[2]))
        end
    end

    isempty(r_values) && error("No data found in $filename")

    println("Loaded potential data from $filename: $(length(r_values)) points")

    return PotentialData(r_values, potential_values)
end

function create_potential_lookup(potential_data::PotentialData, bin_width::Float64, n_bins::Int)::PotentialLookup
    r_min = potential_data.r_values[1]
    r_max = min(potential_data.r_values[end], n_bins * bin_width)

    # Увеличиваем разрешение в 10 раз относительно бинов гистограммы
    n_points = ceil(Int, (r_max - r_min) / (bin_width / 10)) + 1
    r_step = (r_max - r_min) / (n_points - 1)

    lookup_r = [r_min + i * r_step for i in 0:(n_points - 1)]
    lookup_v = [interpolate_potential(potential_data, r) for r in lookup_r]

    println("Created potential lookup table with $(length(lookup_r)) points")
    return PotentialLookup(lookup_r, lookup_v, r_step, r_min, r_max)
end

function interpolate_potential(potential_data::PotentialData, r::Float64)::Float64
    r_values = potential_data.r_values
    potential_values = potential_data.potential_values

    if r <= r_values[1]
        return potential_values[1]
    elseif r >= r_values[end]
        return potential_values[end]
    end

    idx = findfirst(x -> x >= r, r_values)
    if r_values[idx] == r
        return potential_values[idx]
    end

    r1, r2 = r_values[idx - 1], r_values[idx]
    v1, v2 = potential_values[idx - 1], potential_values[idx]

    t = (r - r1) / (r2 - r1)
    return v1 + t * (v2 - v1)
end

@inline function fast_interpolate_potential(r::Float64, lookup::PotentialLookup)::Float64
    if r <= lookup.r_min
        return lookup.lookup_v[1]
    elseif r >= lookup.r_max
        return lookup.lookup_v[end]
    end

    # Быстрый расчет индекса в предвычисленной таблице
    idx = floor(Int, (r - lookup.r_min) / lookup.r_step) + 1
    if idx >= length(lookup.lookup_r)
        return lookup.lookup_v[end]
    elseif idx < 1
        return lookup.lookup_v[1]
    end

    # Линейная интерполяция между предвычисленными значениями
    r1, r2 = lookup.lookup_r[idx], lookup.lookup_r[idx + 1]
    v1, v2 = lookup.lookup_v[idx], lookup.lookup_v[idx + 1]

    t = (r - r1) / (r2 - r1)
    return v1 + t * (v2 - v1)
end

function cache_trajectory(sys_params::SystemParameters)::CachedTrajectory
    println("Caching trajectory for system: $(sys_params.system_name)")
    traj = read_xtc(sys_params)
    n_frames = Int(size(traj)) - 1
    frames = Vector{Frame}(undef, n_frames)
    boxes = Vector{Vector{Float64}}(undef, n_frames)

    for i in 1:n_frames
        frames[i] = deepcopy(read_step(traj, i))
        boxes[i] = lengths(UnitCell(frames[i]))
    end

    println("Cached $(n_frames) frames for system: $(sys_params.system_name)")
    return CachedTrajectory(frames, n_frames, boxes)
end

function read_random_frame_cached(cached_traj::CachedTrajectory, rng::Xoroshiro128Plus)
    frame_id = rand(rng, 1:(cached_traj.n_frames))
    frame = deepcopy(cached_traj.frames[frame_id])
    box = cached_traj.boxes[frame_id]
    return frame, frame_id, box
end

function compute_potential_energy(distance_matrix::Matrix{Float64},
                                  lookup::PotentialLookup,
                                  system_params::SystemParameters)::Float64
    total_energy = 0.0
    n_atoms = system_params.n_atoms
    cutoff = system_params.n_bins * system_params.bin_width

    for i in 1:n_atoms
        for j in (i + 1):n_atoms
            r = distance_matrix[i, j]
            if r > 0.0 && r < cutoff
                pair_energy = fast_interpolate_potential(r, lookup)
                total_energy += pair_energy
            end
        end
    end

    return total_energy
end

function create_magic_reference_data(input::MagicPreComputedInput,
                                     potential_data::PotentialData,
                                     lookup::PotentialLookup)::MagicReferenceData
    nn_params = input.nn_params
    sys_params = input.system_params

    cached_traj = cache_trajectory(sys_params)
    n_frames = cached_traj.n_frames

    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    histograms = Vector{Vector{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    potential_per_frame = Vector{Float64}(undef, n_frames)

    for frame_id in 1:n_frames
        frame = cached_traj.frames[frame_id]

        distance_matrices[frame_id] = build_distance_matrix(frame)
        histograms[frame_id] = zeros(Float64, sys_params.n_bins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], sys_params)

        potential_per_frame[frame_id] = compute_potential_energy(distance_matrices[frame_id], lookup, sys_params)

        g2_matrices[frame_id] = build_g2_matrix(distance_matrices[frame_id], nn_params)
    end

    return MagicReferenceData(distance_matrices, histograms, potential_per_frame, g2_matrices)
end

function magic_single_particle_move!(ref_data::MagicReferenceData,
                                     model::Flux.Chain,
                                     nn_params::NeuralNetParameters,
                                     sys_params::SystemParameters,
                                     lookup::PotentialLookup,
                                     cached_traj::CachedTrajectory,
                                     rng::Xoroshiro128Plus)
    frame, frame_id, box = read_random_frame_cached(cached_traj, rng)
    point_index = rand(rng, 1:(sys_params.n_atoms))

    distance_matrix = ref_data.distance_matrices[frame_id]
    symm1 = ref_data.g2_matrices[frame_id]
    e_nn1_vector = init_system_energies_vector(symm1, model)
    e_nn1 = sum(e_nn1_vector)

    e_pot1 = ref_data.precomputed_energy[frame_id]

    distance_vec1 = distance_matrix[:, point_index]

    dr = sys_params.max_displacement * (rand(rng, Float64, 3) .- 0.5)
    positions(frame)[:, point_index] .+= dr

    point = positions(frame)[:, point_index]
    distance_vec2 = compute_distance_vector(point, positions(frame), box)

    distance_matrix2 = copy(distance_matrix)
    distance_matrix2[point_index, :] = distance_vec2
    distance_matrix2[:, point_index] = distance_vec2

    e_pot2 = compute_potential_energy(distance_matrix2, lookup, sys_params)

    update_mask = get_energies_update_mask(distance_vec1, distance_vec2, nn_params)

    g2_matrix2 = copy(ref_data.g2_matrices[frame_id])
    update_g2_matrix!(g2_matrix2, distance_vec1, distance_vec2, sys_params, nn_params, point_index)

    symm2 = g2_matrix2
    e_nn2_vector = update_system_energies_vector(symm2, model, update_mask, e_nn1_vector)
    e_nn2 = sum(e_nn2_vector)

    positions(frame)[:, point_index] .-= dr

    return (symm1=symm1,
            symm2=symm2,
            Δe_nn=e_nn2 - e_nn1,
            Δe_pot=e_pot2 - e_pot1,
            e_nn1=e_nn1,
            e_pot1=e_pot1,
            e_nn2=e_nn2,
            e_pot2=e_pot2)
end

function magic_all_particle_move!(ref_data::MagicReferenceData,
                                  model::Flux.Chain,
                                  nn_params::NeuralNetParameters,
                                  sys_params::SystemParameters,
                                  lookup::PotentialLookup,
                                  cached_traj::CachedTrajectory,
                                  rng::Xoroshiro128Plus)
    frame, frame_id, box = read_random_frame_cached(cached_traj, rng)

    distance_matrix = ref_data.distance_matrices[frame_id]
    symm1 = ref_data.g2_matrices[frame_id]
    e_nn1_vector = init_system_energies_vector(symm1, model)
    e_nn1 = sum(e_nn1_vector)

    e_pot1 = ref_data.precomputed_energy[frame_id]

    old_coords = copy(positions(frame))

    dr = sys_params.max_displacement * (rand(rng, Float64, 3, sys_params.n_atoms) .- 0.5)
    positions(frame) .+= dr

    for i in 1:(sys_params.n_atoms)
        apply_periodic_boundaries!(frame, box, i)
    end

    distance_matrix2 = build_distance_matrix(frame)
    e_pot2 = compute_potential_energy(distance_matrix2, lookup, sys_params)

    g2_matrix2 = build_g2_matrix(distance_matrix2, nn_params)

    symm2 = g2_matrix2
    e_nn2_vector = init_system_energies_vector(symm2, model)
    e_nn2 = sum(e_nn2_vector)

    positions(frame) .= old_coords

    return (symm1=symm1,
            symm2=symm2,
            Δe_nn=e_nn2 - e_nn1,
            Δe_pot=e_pot2 - e_pot1,
            e_nn1=e_nn1,
            e_pot1=e_pot1,
            e_nn2=e_nn2,
            e_pot2=e_pot2)
end

function make_magic_mc_move!(use_all_particles::Bool,
                             ref_data::MagicReferenceData,
                             model::Chain,
                             nn_params::NeuralNetParameters,
                             sys_params::SystemParameters,
                             lookup::PotentialLookup,
                             cached_traj::CachedTrajectory,
                             rng::Xoroshiro128Plus)
    return use_all_particles ?
           magic_all_particle_move!(ref_data, model, nn_params, sys_params, lookup, cached_traj, rng) :
           magic_single_particle_move!(ref_data, model, nn_params, sys_params, lookup, cached_traj, rng)
end

function run_magic_training_phase!(steps::Int,
                                   batch_size::Int,
                                   use_diff_gradient::Bool,
                                   use_all_particles::Bool,
                                   system_params_list,
                                   ref_data_list,
                                   lookup_list,
                                   cached_traj_list,
                                   model::Chain,
                                   nn_params::NeuralNetParameters,
                                   pretrain_params::PreTrainingParameters,
                                   optimizer;
                                   log_prefix::String="magic_pretraining")
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
                    system_ref_data = ref_data_list[sys_id]
                    system_lookup = lookup_list[sys_id]
                    system_cached_traj = cached_traj_list[sys_id]

                    move = make_magic_mc_move!(use_all_particles,
                                               system_ref_data,
                                               model,
                                               nn_params,
                                               system_params_list[sys_id],
                                               system_lookup,
                                               system_cached_traj,
                                               rng)

                    symm1, symm2 = move.symm1, move.symm2
                    Δe_nn, Δe_pot = move.Δe_nn, move.Δe_pot
                    e_nn2, e_pot2 = move.e_nn2, move.e_pot2
                    n_atoms = system_params_list[sys_id].n_atoms

                    norm = if use_diff_gradient && !use_all_particles
                        1.0
                    else
                        Float64(n_atoms)
                    end

                    batch_gradients[sys_id] = compute_pretraining_gradient!(e_nn2, e_pot2, Δe_nn, Δe_pot,
                                                                            symm1, symm2, model,
                                                                            pretrain_params,
                                                                            use_diff_gradient,
                                                                            norm_factor=norm)

                    accum_mae_diff += abs(Δe_nn - Δe_pot) / n_atoms
                    accum_mae_abs += abs(e_nn2 - e_pot2) / n_atoms
                    count += 1
                end

                flat_grad, grad_restructure = mean_gradient(batch_gradients)
                batch_flat_grad = isnothing(batch_flat_grad) ? flat_grad : batch_flat_grad .+ flat_grad
            end

            batch_flat_grad ./= batch_size
            grad_norm = norm(batch_flat_grad)
            final_grad = grad_restructure(batch_flat_grad)
            update_model!(model, opt_state, final_grad)

            warmup_lr = lr_for_epoch(lr_config, initial_lr, epoch)
            if warmup_lr != lr_state.current_lr
                lr_state.current_lr = warmup_lr
                Flux.adjust!(opt_state, warmup_lr)
            end

            mean_mae_diff = accum_mae_diff / count
            mean_mae_abs = accum_mae_abs / count

            log_pretraining_summary(summary_io, epoch, mean_mae_diff, mean_mae_abs,
                                    grad_norm, lr_state.current_lr)

            plateau_metric = use_diff_gradient ? mean_mae_diff : mean_mae_abs
            step_plateau!(lr_config, lr_state, opt_state, plateau_metric)

            println(@sprintf("Magic PT | %s | %s | Epoch: %4d | Batch Size: %3d | Diff MAE: %8.2f | Abs MAE: %8.2f | LR: %.2e",
                             phase_type, move_type, epoch, batch_size, mean_mae_diff, mean_mae_abs,
                             lr_state.current_lr))

            if epoch % pretrain_params.save_frequency == 0 || epoch == steps
                mode_str = use_diff_gradient ? "diff" : "abs"
                move_str = use_all_particles ? "all" : "single"
                @save joinpath(od, "magic-pt-model-$(mode_str)-$(move_str)-epoch-$(epoch).bson") model
            end

            flush(summary_io)
        end
    finally
        close(summary_io)
    end

    return model, opt_state
end

function magic_pretrain(magic_params::MagicPreTrainingParameters,
                        pretrain_params::PreTrainingParameters,
                        nn_params::NeuralNetParameters,
                        system_params_list::Vector{SystemParameters},
                        model::Chain,
                        optimizer,
                        reference_rdfs::Vector{Vector{Float64}})

    # Загружаем данные о потенциалах
    println("Loading potential data from files...")
    potential_data_list = [load_potential_data(file) for file in magic_params.potential_files]

    # Создаем предвычисленные таблицы значений потенциала для быстрой интерполяции
    println("Creating potential lookup tables...")
    lookup_list = [create_potential_lookup(potential_data_list[i],
                                           system_params_list[i].bin_width,
                                           system_params_list[i].n_bins)
                   for i in 1:length(potential_data_list)]

    # Кэшируем траектории для каждой системы
    println("Caching trajectories...")
    cached_traj_list = [cache_trajectory(system_params) for system_params in system_params_list]

    # Создаем референсные данные для каждой системы
    println("Creating reference data for magic pre-training...")
    ref_inputs = [MagicPreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                  for i in 1:length(system_params_list)]

    ref_data_list = []
    for i in 1:length(ref_inputs)
        println("Processing system $(i)...")
        push!(ref_data_list, create_magic_reference_data(ref_inputs[i], potential_data_list[i], lookup_list[i]))
    end

    println("Starting magic pre-training...")
    model,
    opt_state = run_magic_training_phase!(pretrain_params.steps,
                                          pretrain_params.batch_size,
                                          magic_params.use_diff_gradient,
                                          magic_params.use_all_particles,
                                          system_params_list,
                                          ref_data_list,
                                          lookup_list,
                                          cached_traj_list,
                                          model,
                                          nn_params,
                                          pretrain_params,
                                          optimizer,
                                          log_prefix=pretrain_params.output_prefix)

    # Сохраняем финальную модель
    prefix = pretrain_params.output_prefix
    mode_str = magic_params.use_diff_gradient ? "diff" : "abs"
    move_str = magic_params.use_all_particles ? "all" : "single"
    od = pretrain_params.output_dir
    @save joinpath(od, "$prefix-final-model-$(mode_str)-$(move_str).bson") model
    @save joinpath(od, "$prefix-final-opt-state-$(mode_str)-$(move_str).bson") opt_state

    println("Magic pre-training completed!")
    return model
end
