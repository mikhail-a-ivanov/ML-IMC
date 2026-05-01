using ..ML_IMC

struct CachedTrajectory
    coordinates::Vector{Matrix{Float32}}
    boxes::Vector{Vector{Float32}}
    n_frames::Int
end

struct PretrainingReferenceCache
    trajectory::CachedTrajectory
    distance_matrices::Vector{Matrix{Float32}}
    histograms::Vector{Vector{Float32}}
    g2_matrices::Vector{Matrix{Float32}}
end

function cache_trajectory(sys_params::SystemParameters)::CachedTrajectory
    println("Caching trajectory for system: $(sys_params.system_name)")

    traj = read_xtc(sys_params)
    n_frames = Int(size(traj)) - 1
    coordinates = Vector{Matrix{Float32}}(undef, n_frames)
    boxes = Vector{Vector{Float32}}(undef, n_frames)

    for frame_id in 1:n_frames
        frame = read_step(traj, frame_id)
        coordinates[frame_id] = Matrix{Float32}(positions(frame))
        boxes[frame_id] = Vector{Float32}(lengths(UnitCell(frame)))
    end

    println("Cached $(n_frames) frames for system: $(sys_params.system_name)")
    return CachedTrajectory(coordinates, boxes, n_frames)
end

function sample_cached_frame(cached_traj::CachedTrajectory,
                             rng::Xoroshiro128Plus)
    frame_id = rand(rng, 1:(cached_traj.n_frames))
    return cached_traj.coordinates[frame_id], frame_id, cached_traj.boxes[frame_id]
end

function sample_reference_frame(ref_cache::PretrainingReferenceCache,
                                rng::Xoroshiro128Plus)
    return sample_cached_frame(ref_cache.trajectory, rng)
end

function precompute_reference_cache(nn_params::NeuralNetParameters,
                                    sys_params::SystemParameters)::PretrainingReferenceCache
    cached_traj = cache_trajectory(sys_params)
    n_frames = cached_traj.n_frames

    distance_matrices = Vector{Matrix{Float32}}(undef, n_frames)
    histograms = Vector{Vector{Float32}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float32}}(undef, n_frames)

    for frame_id in 1:n_frames
        coordinates = cached_traj.coordinates[frame_id]
        box = cached_traj.boxes[frame_id]

        distance_matrices[frame_id] = build_distance_matrix(coordinates, box)
        histograms[frame_id] = zeros(Float32, sys_params.n_bins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], sys_params)
        g2_matrices[frame_id] = build_g2_matrix(distance_matrices[frame_id], nn_params)
    end

    return PretrainingReferenceCache(cached_traj, distance_matrices, histograms, g2_matrices)
end

function apply_periodic_boundaries!(coordinates::AbstractMatrix{T},
                                    box::AbstractVector{T},
                                    point_index::Integer) where {T <: AbstractFloat}
    @inbounds for dim in axes(coordinates, 1)
        coord = coordinates[dim, point_index]
        box_length = box[dim]

        if coord < zero(T)
            coordinates[dim, point_index] += box_length
        elseif coord > box_length
            coordinates[dim, point_index] -= box_length
        end
    end

    return coordinates
end

function apply_periodic_boundaries!(coordinates::AbstractMatrix{T},
                                    box::AbstractVector{T}) where {T <: AbstractFloat}
    @inbounds for point_index in axes(coordinates, 2)
        apply_periodic_boundaries!(coordinates, box, point_index)
    end

    return coordinates
end

function apply_periodic_boundaries!(position::AbstractVector{T},
                                    box::AbstractVector{T}) where {T <: AbstractFloat}
    @inbounds for dim in eachindex(position, box)
        coord = position[dim]
        box_length = box[dim]

        if coord < zero(T)
            position[dim] += box_length
        elseif coord > box_length
            position[dim] -= box_length
        end
    end

    return position
end

function random_displaced_position(coordinates::AbstractMatrix{T},
                                   point_index::Integer,
                                   box::AbstractVector{T},
                                   max_displacement::T,
                                   rng::Xoroshiro128Plus)::Vector{T} where {T <: AbstractFloat}
    position = Vector{T}(undef, size(coordinates, 1))

    @inbounds for dim in axes(coordinates, 1)
        position[dim] = coordinates[dim, point_index] + max_displacement * (rand(rng, T) - T(0.5))
    end

    apply_periodic_boundaries!(position, box)
    return position
end

function random_displaced_coordinates(coordinates::AbstractMatrix{T},
                                      box::AbstractVector{T},
                                      max_displacement::T,
                                      rng::Xoroshiro128Plus)::Matrix{T} where {T <: AbstractFloat}
    displaced_coordinates = copy(coordinates)

    @inbounds for atom_index in axes(displaced_coordinates, 2)
        for dim in axes(displaced_coordinates, 1)
            displaced_coordinates[dim, atom_index] += max_displacement * (rand(rng, T) - T(0.5))
        end
    end

    apply_periodic_boundaries!(displaced_coordinates, box)
    return displaced_coordinates
end

function compute_pretraining_gradient!(e_nn::T, e_ref::T, Δe_nn::T, Δe_ref::T,
                                       symm1::AbstractMatrix{T}, symm2::AbstractMatrix{T},
                                       model::Chain, pretrain_params::PreTrainingParameters,
                                       use_diff_gradient::Bool;
                                       norm_factor::T=T(1))::Any where {T <: AbstractFloat}
    model_params = Flux.trainables(model)
    mp, _ = Flux.destructure(model_params)
    reg_gradient = @. mp * T(2) * T(pretrain_params.regularization)

    if !use_diff_gradient
        grad_final = compute_energy_gradients(symm2, model)
        diff = (e_nn - e_ref) / norm_factor
        gradient_scale = pretrain_params.gradient_type == "mae" ? sign(diff) : T(2) * diff
        flat_grad, restructure = Flux.destructure(grad_final)
        loss_gradient = @. gradient_scale * flat_grad / norm_factor
        return restructure(loss_gradient + reg_gradient)
    else
        grad1 = compute_energy_gradients(symm1, model)
        grad2 = compute_energy_gradients(symm2, model)
        diff = (Δe_nn - Δe_ref) / norm_factor
        gradient_scale = pretrain_params.gradient_type == "mae" ? sign(diff) : T(2) * diff
        flat_grad1, restructure = Flux.destructure(grad1)
        flat_grad2, _ = Flux.destructure(grad2)
        loss_gradient = @. gradient_scale * (flat_grad2 - flat_grad1) / norm_factor
        return restructure(loss_gradient + reg_gradient)
    end
end

function log_pretraining_summary(io::IO, epoch::Int, mean_diff_mae, mean_abs_mae,
                                 grad_norm, lr)
    println(io, @sprintf("%d,%.17e,%.17e,%.17e,%.17e", epoch, mean_diff_mae, mean_abs_mae,
                         grad_norm, lr))
end

function mean_gradient(batch_gradients::Vector{Any})
    n = length(batch_gradients)
    first_flat_grad, grad_restructure = Flux.destructure(batch_gradients[1])
    T = eltype(first_flat_grad)
    for grad in batch_gradients[2:end]
        flat_grad, _ = Flux.destructure(grad)
        first_flat_grad .+= flat_grad
    end
    first_flat_grad ./= T(n)
    return first_flat_grad, grad_restructure
end
