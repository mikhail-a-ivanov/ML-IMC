using ..ML_IMC

struct MonteCarloAverages
    descriptor::Vector{Float64}
    energies::Vector{Float64}
    cross_accumulators::Union{Nothing, Matrix{Float64}}
    symmetry_matrix_accumulator::Union{Nothing, Matrix{Float64}}
    acceptance_ratio::Float64
    system_params::SystemParameters
    step_size::Float64
end

struct MonteCarloSampleInput
    global_params::GlobalParameters
    mc_params::MonteCarloParameters
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    model::Chain
end

function update_distance_histogram!(distance_matrix::Matrix{T},
                                    histogram::Vector{T},
                                    system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_atoms = system_params.n_atoms
    bin_width = system_params.bin_width
    n_bins = system_params.n_bins

    @inbounds for i in 1:n_atoms
        @fastmath for j in 1:(i - 1)
            bin_index = floor(Int, 1 + distance_matrix[i, j] / bin_width)
            if bin_index <= n_bins
                histogram[bin_index] += 1
            end
        end
    end

    return histogram
end

function update_distance_histogram_vectors!(histogram::Vector{T},
                                            old_distances::Vector{T},
                                            new_distances::Vector{T},
                                            system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_atoms = system_params.n_atoms
    bin_width = system_params.bin_width
    n_bins = system_params.n_bins

    @inbounds @fastmath for i in 1:n_atoms
        old_bin = floor(Int, 1 + old_distances[i] / bin_width)
        if old_bin <= n_bins
            histogram[old_bin] -= 1
        end

        new_bin = floor(Int, 1 + new_distances[i] / bin_width)
        if new_bin <= n_bins
            histogram[new_bin] += 1
        end
    end

    return histogram
end

function normalize_hist_to_rdf!(histogram::AbstractVector{T},
                                system_params::SystemParameters,
                                box::AbstractVector{T})::AbstractVector{T} where {T <: AbstractFloat}
    box_volume = prod(box)
    n_pairs = system_params.n_atoms * (system_params.n_atoms - 1) ÷ 2

    bin_width = system_params.bin_width
    bins = [(i * bin_width) for i in 1:(system_params.n_bins)]
    shell_volumes = [4 * π * system_params.bin_width * bins[i]^2 for i in eachindex(bins)]

    normalization_factors = fill(box_volume / n_pairs, system_params.n_bins) ./ shell_volumes

    histogram .*= normalization_factors

    return histogram
end

function collect_system_averages(outputs::Vector{MonteCarloAverages},
                                 reference_rdfs::Any,
                                 system_params_list::Vector{SystemParameters},
                                 global_params::GlobalParameters,
                                 nn_params::Union{NeuralNetParameters, Nothing},
                                 model::Union{Flux.Chain, Nothing}, lr::Float64,
                                 epoch, mc_steps)::Tuple{Vector{MonteCarloAverages}, Vector{Float64}}
    total_loss_sse::Float64 = 0.0
    total_loss_mse::Float64 = 0.0
    total_loss_rmse::Float64 = 0.0
    total_loss_mae::Float64 = 0.0

    system_outputs::Vector{MonteCarloAverages} = Vector{MonteCarloAverages}()
    system_losses::Vector{Float64} = Vector{Float64}()

    println("| System          | Acc.Ratio | Avg.Displ.(Å) | SSE        | MSE        | RMSE     | MAE      |")

    for (system_idx, system_params) in enumerate(system_params_list)
        descriptors::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
        energies::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
        acceptance_ratios::Vector{Float64} = Vector{Float64}()
        max_displacements::Vector{Float64} = Vector{Float64}()

        cross_accumulators::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}()
        symm_func_accumulators::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}()

        for output in outputs
            if system_params.system_name == output.system_params.system_name
                push!(descriptors, output.descriptor)
                push!(energies, output.energies)
                push!(acceptance_ratios, output.acceptance_ratio)
                push!(max_displacements, output.step_size)

                if global_params.mode == "training"
                    push!(cross_accumulators, output.cross_accumulators)
                    push!(symm_func_accumulators, output.symmetry_matrix_accumulator)
                end
            end
        end

        avg_descriptor::Vector{Float64} = mean(descriptors)
        avg_energies::Vector{Float64} = mean(energies)
        avg_acceptance::Float64 = mean(acceptance_ratios)
        avg_displacement::Float64 = mean(max_displacements)

        avg_cross_acc::Union{Matrix{Float64}, Nothing} = nothing
        avg_symm_func::Union{Matrix{Float64}, Nothing} = nothing

        if global_params.mode == "training"
            avg_cross_acc = mean(cross_accumulators)
            avg_symm_func = mean(symm_func_accumulators)
        end

        system_output = MonteCarloAverages(avg_descriptor,
                                           avg_energies,
                                           avg_cross_acc,
                                           avg_symm_func,
                                           avg_acceptance,
                                           system_params,
                                           avg_displacement)

        if global_params.mode != "training"
            println("System $(system_params.system_name):")
            println("    Acceptance ratio: ", round(avg_acceptance; digits=4))
            println("    Avg. displacement: ", round(avg_displacement; digits=4))
            println()
        end

        if global_params.mode == "training"
            system_loss_mae, system_loss_sse, system_loss_mse,
            system_loss_rmse = compute_training_loss(system_output.descriptor,
                                                     reference_rdfs[system_idx],
                                                     model,
                                                     nn_params,
                                                     global_params.output_dir)

            println(@sprintf("| %-15s | %9.4f | %13.4f | %10.4f | %9.4e | %8.4f | %8.4f |",
                             system_params.system_name,
                             avg_acceptance,
                             avg_displacement,
                             system_loss_sse,
                             system_loss_mse,
                             system_loss_rmse,
                             system_loss_mae))

            total_loss_sse += system_loss_sse
            total_loss_mse += system_loss_mse
            total_loss_rmse += system_loss_rmse
            total_loss_mae += system_loss_mae
            push!(system_losses, system_loss_mae)
        end

        push!(system_outputs, system_output)
    end

    if global_params.mode == "training"
        total_loss_sse /= length(system_params_list)
        total_loss_mse /= length(system_params_list)
        total_loss_rmse /= length(system_params_list)
        total_loss_mae /= length(system_params_list)
        println()
        println(@sprintf("Epoch: %d | Steps: %d | SSE: %.5f | MSE: %.4e | RMSE: %.8f | MAE: %.8f | LR: %.2e",
                         epoch, mc_steps, total_loss_sse, total_loss_mse, total_loss_rmse, total_loss_mae, lr))
    end

    od = global_params.output_dir
    try
        open(joinpath(od, "avg_sse_loss.dat"), "a") do io
            println(io, round(total_loss_sse; digits=8))
        end
    catch e
        @warn "Failed to write to log file: avg_sse_loss.dat" exception=e
    end

    try
        open(joinpath(od, "avg_mae_loss.dat"), "a") do io
            println(io, round(total_loss_mae; digits=8))
        end
    catch e
        @warn "Failed to write to log file: avg_mae_loss.dat" exception=e
    end

    return (system_outputs, system_losses)
end

function apply_periodic_boundaries!(frame::Frame,
                                    box::AbstractVector{T},
                                    point_index::Integer) where {T <: AbstractFloat}
    pos = positions(frame)

    @inbounds for dim in 1:3
        coord = pos[dim, point_index]
        box_length = box[dim]

        if coord < zero(T)
            pos[dim, point_index] += box_length
        elseif coord > box_length
            pos[dim, point_index] -= box_length
        end
    end

    return frame
end

function mcmove!(mc_arrays::Tuple{Frame, Matrix{T}, Matrix{T}},
                 current_energy::T,
                 energy_prev_vec::Vector{T},
                 model::Flux.Chain,
                 nn_params::NeuralNetParameters,
                 system_params::SystemParameters,
                 box::Vector{T},
                 rng::Xoroshiro128Plus,
                 step_size::T)::Tuple{Tuple{Frame, Matrix{T}, Matrix{T}},
                                      T,
                                      Vector{T},
                                      Int} where {T <: AbstractFloat}
    frame, distance_matrix, g2_matrix = mc_arrays

    particle_index = rand(rng, 1:(system_params.n_atoms))
    distances_orig = @view distance_matrix[:, particle_index]
    energy_orig = copy(current_energy)

    displacement = step_size * (rand(rng, T, 3) .- 0.5)
    positions(frame)[:, particle_index] .+= displacement
    apply_periodic_boundaries!(frame, box, particle_index)

    new_position = positions(frame)[:, particle_index]
    distances_new = compute_distance_vector(new_position, positions(frame), box)

    indexes_for_update = get_energies_update_mask(distances_new, nn_params)

    g2_matrix_new = copy(g2_matrix)
    update_g2_matrix!(g2_matrix_new, distances_orig, distances_new, system_params, nn_params, particle_index)

    energy_vector_new = update_system_energies_vector(g2_matrix_new, model, indexes_for_update, energy_prev_vec)
    energy_new = sum(energy_vector_new)

    accepted = 0
    if rand(rng, T) < exp(-((energy_new - energy_orig) * system_params.beta))
        accepted = 1
        current_energy = energy_new
        distance_matrix[particle_index, :] = distances_new
        distance_matrix[:, particle_index] = distances_new
        return ((frame, distance_matrix, g2_matrix_new),
                current_energy, energy_vector_new, accepted)
    else
        positions(frame)[:, particle_index] .-= displacement
        apply_periodic_boundaries!(frame, box, particle_index)
        return ((frame, distance_matrix, g2_matrix),
                current_energy, energy_prev_vec, accepted)
    end
end

function mcsample!(input::MonteCarloSampleInput)::MonteCarloAverages
    model = input.model
    global_params = input.global_params
    mc_params = input.mc_params
    nn_params = input.nn_params
    system_params = input.system_params

    step_size::Float64 = copy(system_params.max_displacement)

    worker_id::Int = nprocs() == 1 ? myid() : myid() - 1
    worker_id_str::String = lpad(worker_id, 3, "0")
    od = global_params.output_dir

    traj_file::String = joinpath(od, "mctraj-p$(worker_id_str).xtc")
    pdb_file::String = joinpath(od, "confin-p$(worker_id_str).pdb")

    rng::Xoroshiro128Plus = Xoroshiro128Plus()

    frame::Frame = if global_params.mode == "training"
        trajectory = read_xtc(system_params)
        n_frames::Int = Int(size(trajectory)) - 1
        frame_id::Int = rand(rng, 1:n_frames)
        deepcopy(read_step(trajectory, frame_id))
    else
        pdb = read_pdb(system_params)
        deepcopy(read_step(pdb, 0))
    end

    box::Vector{Float64} = lengths(UnitCell(frame))

    if global_params.output_mode == "verbose"
        write_trajectory(positions(frame), box, system_params, traj_file, 'w')
        write_trajectory(positions(frame), box, system_params, pdb_file, 'w')
    end

    total_points::Int = round(Int, mc_params.steps / mc_params.output_frequency)
    production_points::Int = round(Int, (mc_params.steps - mc_params.equilibration_steps) / mc_params.output_frequency)

    distance_matrix::Matrix{Float64} = build_distance_matrix(frame)
    g2_matrix::Matrix{Float64} = build_g2_matrix(distance_matrix, nn_params)

    mc_arrays::Tuple = (frame, distance_matrix, g2_matrix)

    hist_accumulator::Vector{Float64} = zeros(Float64, system_params.n_bins)

    if global_params.mode == "training"
        hist::Vector{Float64} = zeros(Float64, system_params.n_bins)
        g2_accumulator::Matrix{Float64} = zeros(size(g2_matrix))
        cross_accumulators::Matrix{Float64} = initialize_cross_accumulators(system_params, model)
    end

    symm_func_matrix::Matrix{Float64} = g2_matrix
    energy_vector::Vector{Float64} = init_system_energies_vector(symm_func_matrix, model)
    total_energy::Float64 = sum(energy_vector)
    energies::Vector{Float64} = zeros(total_points + 1)
    energies[1] = total_energy

    accepted_total::Int = 0
    accepted_intermediate::Int = 0

    for step in 1:(mc_params.steps)
        mc_arrays, total_energy, energy_vector,
        accepted = mcmove!(mc_arrays, total_energy, energy_vector, model,
                           nn_params, system_params, box, rng, step_size)
        accepted_total += accepted
        accepted_intermediate += accepted

        if mc_params.step_adjust_frequency > 0 &&
           step % mc_params.step_adjust_frequency == 0 &&
           step < mc_params.equilibration_steps
            step_size = adjust_monte_carlo_step!(step_size, system_params, box, mc_params, accepted_intermediate)
            accepted_intermediate = 0
        end

        if step % mc_params.output_frequency == 0
            energies[Int(step / mc_params.output_frequency) + 1] = total_energy
        end

        if global_params.output_mode == "verbose" &&
           step % mc_params.trajectory_output_frequency == 0
            write_trajectory(positions(mc_arrays[1]), box, system_params, traj_file, 'a')
        end

        if step % mc_params.output_frequency == 0 && step > mc_params.equilibration_steps
            frame, distance_matrix, g2_matrix = mc_arrays

            if global_params.mode == "training"
                hist = update_distance_histogram!(distance_matrix, hist, system_params)
                hist_accumulator .+= hist
                g2_accumulator .+= g2_matrix

                normalize_hist_to_rdf!(hist, system_params, box)
                update_cross_accumulators!(cross_accumulators, g2_matrix, hist, model)
                hist = zeros(Float64, system_params.n_bins)
            else
                hist_accumulator = update_distance_histogram!(distance_matrix, hist_accumulator, system_params)
            end
        end
    end

    acceptance_ratio::Float64 = accepted_total / mc_params.steps

    if global_params.mode == "training"
        cross_accumulators ./= production_points
        g2_accumulator ./= production_points
        symm_func_accumulator = g2_accumulator
    end

    hist_accumulator ./= production_points
    normalize_hist_to_rdf!(hist_accumulator, system_params, box)

    if global_params.mode == "training"
        return MonteCarloAverages(hist_accumulator,
                                  energies,
                                  cross_accumulators,
                                  symm_func_accumulator,
                                  acceptance_ratio,
                                  system_params,
                                  step_size)
    else
        return MonteCarloAverages(hist_accumulator,
                                  energies,
                                  nothing,
                                  nothing,
                                  acceptance_ratio,
                                  system_params,
                                  step_size)
    end
end
