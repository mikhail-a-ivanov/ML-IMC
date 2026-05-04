using ..ML_IMC

struct MonteCarloAverages
    rdf::Vector{Float32}
    energies::Vector{Float32}
    cross_accumulators::Union{Nothing, Matrix{Float32}}
    mean_flat_grad::Union{Nothing, Vector{Float32}}
    acceptance_ratio::Float32
    system_params::SystemParameters
    step_size::Float32
end

struct MonteCarloSampleInput
    global_params::GlobalParameters
    mc_params::MonteCarloParameters
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    model_weights::Vector{Float32}
end

function update_distance_histogram!(distance_matrix::Matrix{T},
                                    histogram::Vector{T},
                                    system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_atoms = system_params.n_atoms
    bin_width = T(system_params.bin_width)
    n_bins = system_params.n_bins

    @inbounds for i in 1:n_atoms
        @fastmath for j in 1:(i - 1)
            bin_index = floor(Int, 1 + distance_matrix[i, j] / bin_width)
            if bin_index <= n_bins
                histogram[bin_index] += one(T)
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
    bin_width = T(system_params.bin_width)
    n_bins = system_params.n_bins

    @inbounds @fastmath for i in 1:n_atoms
        old_bin = floor(Int, 1 + old_distances[i] / bin_width)
        if old_bin <= n_bins
            histogram[old_bin] -= one(T)
        end

        new_bin = floor(Int, 1 + new_distances[i] / bin_width)
        if new_bin <= n_bins
            histogram[new_bin] += one(T)
        end
    end

    return histogram
end

function compute_rdf_normalization_factors(system_params::SystemParameters,
                                           box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    box_volume = prod(box)
    n_pairs = T(system_params.n_atoms * (system_params.n_atoms - 1) ÷ 2)
    bin_width = T(system_params.bin_width)
    shell_volume_factor = T(4) * T(π) / T(3)

    factors = Vector{T}(undef, system_params.n_bins)
    @inbounds for i in 1:(system_params.n_bins)
        shell_volume = shell_volume_factor *
                       ((T(i) * bin_width)^3 - ((T(i) - one(T)) * bin_width)^3)
        factors[i] = box_volume / (n_pairs * shell_volume)
    end
    return factors
end

function collect_system_averages(outputs::Vector{MonteCarloAverages},
                                 rdf_targets::Union{Nothing, Vector{Vector{Float32}}},
                                 system_params_list::Vector{SystemParameters},
                                 global_params::GlobalParameters)::Tuple{Vector{MonteCarloAverages}, Vector{Float32}}
    total_loss_mae::Float32 = 0.0f0

    system_outputs::Vector{MonteCarloAverages} = Vector{MonteCarloAverages}()
    system_losses::Vector{Float32} = Vector{Float32}()

    println("| System          | Acc.Ratio | Avg.Displ.(Å) | MAE       |")

    for (system_idx, system_params) in enumerate(system_params_list)
        rdfs::Vector{Vector{Float32}} = Vector{Vector{Float32}}()
        energies::Vector{Vector{Float32}} = Vector{Vector{Float32}}()
        acceptance_ratios::Vector{Float32} = Vector{Float32}()
        max_displacements::Vector{Float32} = Vector{Float32}()

        cross_accumulators::Vector{Matrix{Float32}} = Vector{Matrix{Float32}}()
        flat_grad_accumulators::Vector{Vector{Float32}} = Vector{Vector{Float32}}()

        for output in outputs
            if system_params.system_name == output.system_params.system_name
                push!(rdfs, output.rdf)
                push!(energies, output.energies)
                push!(acceptance_ratios, output.acceptance_ratio)
                push!(max_displacements, output.step_size)

                if global_params.mode == "training"
                    push!(cross_accumulators, output.cross_accumulators::Matrix{Float32})
                    push!(flat_grad_accumulators, output.mean_flat_grad::Vector{Float32})
                end
            end
        end

        avg_rdf::Vector{Float32} = mean(rdfs)
        avg_energies::Vector{Float32} = mean(energies)
        avg_acceptance::Float32 = mean(acceptance_ratios)
        avg_displacement::Float32 = mean(max_displacements)

        avg_cross_acc::Union{Matrix{Float32}, Nothing} = nothing
        avg_flat_grad::Union{Vector{Float32}, Nothing} = nothing

        if global_params.mode == "training"
            avg_cross_acc = mean(cross_accumulators)
            avg_flat_grad = mean(flat_grad_accumulators)
        end

        system_output = MonteCarloAverages(avg_rdf,
                                           avg_energies,
                                           avg_cross_acc,
                                           avg_flat_grad,
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
            isnothing(rdf_targets) && throw(ArgumentError("rdf_targets are required in training mode"))

            system_loss_mae,
            system_loss_rmse = compute_training_loss(system_output.rdf,
                                                     rdf_targets[system_idx])

            println(@sprintf("| %-15s | %9.4f | %13.4f | %.3e |",
                             system_params.system_name,
                             avg_acceptance,
                             avg_displacement,
                             system_loss_mae))

            total_loss_mae += system_loss_mae
            push!(system_losses, system_loss_mae)
        end

        push!(system_outputs, system_output)
    end

    if global_params.mode == "training"
        total_loss_mae /= Float32(length(system_params_list))
    end

    return (system_outputs, system_losses)
end

function mcmove!(coordinates::AbstractMatrix{T},
                 distance_matrix::Matrix{T},
                 g2_matrix::Matrix{T},
                 current_energy::T,
                 energy_vector::Vector{T},
                 model::Flux.Chain,
                 nn_params::NeuralNetParameters,
                 system_params::SystemParameters,
                 box::Vector{T},
                 rng::Xoroshiro128Plus,
                 step_size::T,
                 workspace::MonteCarloWorkspace{T})::Tuple{T, Int} where {T <: AbstractFloat}
    n_atoms = system_params.n_atoms
    max_cutoff = nn_params.max_distance_cutoff
    n_g2 = length(nn_params.g2_functions)

    particle_index = rand(rng, 1:n_atoms)
    energy_orig = current_energy

    @inbounds for dim in 1:3
        workspace.displacement[dim] = step_size * (rand(rng, T) - T(0.5))
        coordinates[dim, particle_index] += workspace.displacement[dim]
    end
    apply_periodic_boundaries!(coordinates, box, particle_index)

    old_distances = @view distance_matrix[:, particle_index]
    compute_distance_vector_from_column!(workspace.new_distances, coordinates, particle_index, box)

    empty!(workspace.affected_indices)
    push!(workspace.affected_indices, particle_index)
    @inbounds for i in 1:n_atoms
        i == particle_index && continue
        if old_distances[i] < max_cutoff || workspace.new_distances[i] < max_cutoff
            push!(workspace.affected_indices, i)
        end
    end
    n_affected = length(workspace.affected_indices)

    compute_changed_g2_rows!(workspace.g2_rows_scratch, workspace.affected_indices, n_affected,
                             g2_matrix, old_distances, workspace.new_distances, nn_params, particle_index)

    proposed_energies = vec(model(@view workspace.g2_rows_scratch[:, 1:n_affected]))

    delta_energy = zero(T)
    @inbounds for k in 1:n_affected
        i = workspace.affected_indices[k]
        delta_energy += proposed_energies[k] - energy_vector[i]
    end
    energy_new = current_energy + delta_energy

    accepted = 0
    if rand(rng, T) < exp(-((energy_new - energy_orig) * system_params.beta))
        accepted = 1

        @inbounds for i in 1:n_atoms
            distance_matrix[particle_index, i] = workspace.new_distances[i]
            distance_matrix[i, particle_index] = workspace.new_distances[i]
        end

        @inbounds for k in 1:n_affected
            atom_i = workspace.affected_indices[k]
            for g in 1:n_g2
                g2_matrix[g, atom_i] = workspace.g2_rows_scratch[g, k]
            end
        end

        @inbounds for k in 1:n_affected
            energy_vector[workspace.affected_indices[k]] = proposed_energies[k]
        end
        current_energy = energy_new
    else
        @inbounds for dim in 1:3
            coordinates[dim, particle_index] -= workspace.displacement[dim]
        end
        apply_periodic_boundaries!(coordinates, box, particle_index)
    end

    return (current_energy, accepted)
end

function mcsample!(input::MonteCarloSampleInput)::MonteCarloAverages
    global_params = input.global_params
    mc_params = input.mc_params
    nn_params = input.nn_params
    system_params = input.system_params
    model_weights = input.model_weights

    # Rebuild model from weights on the worker
    # This avoids serialization issues with the model structure itself
    model = model_init(nn_params)
    _, rebuild = Flux.destructure(model)
    model = f32(rebuild(model_weights))

    step_size::Float32 = copy(system_params.max_displacement)

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

    coordinates::Matrix{Float32} = Matrix{Float32}(positions(frame))
    box::Vector{Float32} = Vector{Float32}(lengths(UnitCell(frame)))

    if global_params.output_mode == "verbose"
        write_trajectory(coordinates, box, system_params, traj_file, 'w')
        write_trajectory(coordinates, box, system_params, pdb_file, 'w')
    end

    total_steps::Int = mc_params.equilibration_steps + mc_params.steps
    total_output_points::Int = round(Int, Float32(total_steps) / Float32(mc_params.output_frequency))

    distance_matrix::Matrix{Float32} = build_distance_matrix(coordinates, box)
    g2_matrix::Matrix{Float32} = build_g2_matrix(distance_matrix, nn_params)

    n_params::Int = length(model_weights)
    workspace = MonteCarloWorkspace{Float32}(system_params.n_atoms, length(nn_params.g2_functions), n_params)

    hist_accumulator::Vector{Float32} = zeros(Float32, system_params.n_bins)
    rdf_norm_factors::Vector{Float32} = compute_rdf_normalization_factors(system_params, box)

    if global_params.mode == "training"
        hist::Vector{Float32} = zeros(Float32, system_params.n_bins)
        cross_accumulators::Matrix{Float32} = initialize_cross_accumulators(system_params, model)
        flat_grad_accumulator::Vector{Float32} = zeros(Float32, n_params)
    end

    energy_vector::Vector{Float32} = init_system_energies_vector(g2_matrix, model)
    total_energy::Float32 = sum(energy_vector)
    energies::Vector{Float32} = zeros(Float32, total_output_points + 1)
    energies[1] = total_energy

    accepted_total::Int = 0
    accepted_intermediate::Int = 0
    collected_samples::Int = 0

    for step in 1:total_steps
        total_energy,
        accepted = mcmove!(coordinates, distance_matrix, g2_matrix,
                           total_energy, energy_vector, model,
                           nn_params, system_params, box, rng, step_size, workspace)
        accepted_total += accepted
        accepted_intermediate += accepted

        if mc_params.step_adjust_frequency > 0 &&
           step % mc_params.step_adjust_frequency == 0 &&
           step < mc_params.equilibration_steps
            step_size = adjust_monte_carlo_step!(step_size, system_params, box, mc_params, accepted_intermediate)
            accepted_intermediate = 0
        end

        if step % mc_params.output_frequency == 0
            energies[(step ÷ mc_params.output_frequency) + 1] = total_energy
        end

        if global_params.output_mode == "verbose" &&
           step % mc_params.trajectory_output_frequency == 0
            write_trajectory(coordinates, box, system_params, traj_file, 'a')
        end

        if step % mc_params.output_frequency == 0 && step > mc_params.equilibration_steps
            collected_samples += 1
            if global_params.mode == "training"
                hist = update_distance_histogram!(distance_matrix, hist, system_params)
                hist_accumulator .+= hist

                update_cross_accumulators!(cross_accumulators, g2_matrix, hist, model, workspace.flat_grad_buffer)
                flat_grad_accumulator .+= workspace.flat_grad_buffer
                fill!(hist, 0.0f0)
            else
                hist_accumulator = update_distance_histogram!(distance_matrix, hist_accumulator, system_params)
            end
        end
    end

    acceptance_ratio::Float32 = Float32(accepted_total) / Float32(total_steps)

    if collected_samples == 0
        error("No production samples collected in mcsample!. Check configuration: steps=$(mc_params.steps), equilibration=$(mc_params.equilibration_steps), output_frequency=$(mc_params.output_frequency).")
    end

    inv_n_samples = one(Float32) / Float32(collected_samples)

    if global_params.mode == "training"
        cross_accumulators .*= rdf_norm_factors .* inv_n_samples
        flat_grad_accumulator .*= inv_n_samples
    end

    hist_accumulator .*= rdf_norm_factors .* inv_n_samples

    if global_params.mode == "training"
        return MonteCarloAverages(hist_accumulator,
                                  energies,
                                  cross_accumulators,
                                  flat_grad_accumulator,
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
