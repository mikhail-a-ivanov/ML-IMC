using ..ML_IMC

function distance_cutoff(distance::Float64, r_cutoff::Float64)::Float64
    if distance > r_cutoff
        return 0.0
    else
        return (0.5 * (cos(π * distance / r_cutoff) + 1.0))
    end
end

function compute_g2_element(distance::T,
                            η::T,
                            r_cutoff::T,
                            r_shift::T,
                            norm::T)::T where {T <: AbstractFloat}
    distance <= zero(T) && return zero(T)

    shifted_distance = distance - r_shift
    exponential_term = exp(-η * shifted_distance^2)
    cutoff_term = distance_cutoff(distance, r_cutoff)

    return exponential_term * cutoff_term / norm
end

function compute_g2(distances::AbstractVector{T},
                    η::T,
                    r_cutoff::T,
                    r_shift::T, norm::T)::T where {T <: AbstractFloat}
    acc = zero(T)
    @simd for i in eachindex(distances)
        @inbounds acc += compute_g2_element(distances[i], η, r_cutoff, r_shift, norm)
    end
    return acc
end

function build_g2_matrix(distance_matrix::AbstractMatrix{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g2_functions = length(nn_params.g2_functions)
    g2_matrix = Matrix{T}(undef, n_g2_functions, n_atoms)

    for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]
        for (j, g2_func) in enumerate(nn_params.g2_functions)
            g2_matrix[j, i] = compute_g2(distance_vector,
                                         g2_func.eta,
                                         g2_func.rcutoff,
                                         g2_func.rshift, g2_func.norm)
        end
    end

    return nn_params.symm_function_scaling == one(T) ? g2_matrix :
           g2_matrix .* nn_params.symm_function_scaling
end

function update_g2_matrix!(g2_matrix::AbstractMatrix{T},
                           distance_vector1::AbstractVector{T},
                           distance_vector2::AbstractVector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           point_index::Integer)::AbstractMatrix{T} where {T <: AbstractFloat}
    scaling = nn_params.symm_function_scaling

    @inbounds for (j, g2_func) in enumerate(nn_params.g2_functions)
        g2_matrix[j, point_index] = compute_g2(distance_vector2,
                                               g2_func.eta,
                                               g2_func.rcutoff,
                                               g2_func.rshift, g2_func.norm) * scaling
    end

    @inbounds for i in 1:(system_params.n_atoms)
        i == point_index && continue

        for (j, g2_func) in enumerate(nn_params.g2_functions)
            r_cutoff = g2_func.rcutoff
            dist1, dist2 = distance_vector1[i], distance_vector2[i]

            if (zero(T) < dist1 < r_cutoff) || (zero(T) < dist2 < r_cutoff)
                δg2 = compute_g2_element(dist2, g2_func.eta, r_cutoff, g2_func.rshift, g2_func.norm) -
                      compute_g2_element(dist1, g2_func.eta, r_cutoff, g2_func.rshift, g2_func.norm)
                g2_matrix[j, i] += δg2 * scaling
            end
        end
    end

    return g2_matrix
end

function compute_changed_g2_rows!(g2_scratch::AbstractMatrix{T},
                                  affected_indices::AbstractVector{Int},
                                  n_affected::Int,
                                  g2_matrix::AbstractMatrix{T},
                                  old_distances::AbstractVector{T},
                                  new_distances::AbstractVector{T},
                                  nn_params::NeuralNetParameters,
                                  point_index::Int)::Nothing where {T <: AbstractFloat}
    scaling = nn_params.symm_function_scaling

    @inbounds for (j, g2_func) in enumerate(nn_params.g2_functions)
        g2_scratch[j, 1] = compute_g2(new_distances,
                                      g2_func.eta,
                                      g2_func.rcutoff,
                                      g2_func.rshift,
                                      g2_func.norm) * scaling
    end

    @inbounds for k in 2:n_affected
        i = affected_indices[k]
        dist1 = old_distances[i]
        dist2 = new_distances[i]
        for (j, g2_func) in enumerate(nn_params.g2_functions)
            r_cutoff = g2_func.rcutoff
            delta = compute_g2_element(dist2, g2_func.eta, r_cutoff,
                                       g2_func.rshift, g2_func.norm) -
                    compute_g2_element(dist1, g2_func.eta, r_cutoff,
                                       g2_func.rshift, g2_func.norm)
            g2_scratch[j, k] = g2_matrix[j, i] + delta * scaling
        end
    end

    return nothing
end
