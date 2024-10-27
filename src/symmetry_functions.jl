using ..ML_IMC

function combine_symmetry_matrices(g2_matrix, g3_matrix, g9_matrix) # NOTE: better no types
    if isempty(g3_matrix) && isempty(g9_matrix)
        return g2_matrix
    end

    matrices = [g2_matrix, g3_matrix, g9_matrix]
    return hcat(filter(!isempty, matrices)...)
end

function compute_g2_element(distance::T,
                            η::T,
                            r_cutoff::T,
                            r_shift::T)::T where {T <: AbstractFloat}
    distance <= zero(T) && return zero(T)

    shifted_distance = distance - r_shift
    exponential_term = exp(-η * shifted_distance^2)
    cutoff_term = distance_cutoff(distance, r_cutoff)

    return exponential_term * cutoff_term
end

function compute_g2(distances::AbstractVector{T},
                    η::T,
                    r_cutoff::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    acc = zero(T)
    @simd for i in eachindex(distances)
        @inbounds acc += compute_g2_element(distances[i], η, r_cutoff, r_shift)
    end
    return acc
end

function build_g2_matrix(distance_matrix::AbstractMatrix{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g2_functions = length(nn_params.g2_functions)
    g2_matrix = Matrix{T}(undef, n_atoms, n_g2_functions)

    for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]
        for (j, g2_func) in enumerate(nn_params.g2_functions)
            g2_matrix[i, j] = compute_g2(distance_vector,
                                         g2_func.eta,
                                         g2_func.rcutoff,
                                         g2_func.rshift)
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

    # Update displaced particle row
    @inbounds for (j, g2_func) in enumerate(nn_params.g2_functions)
        g2_matrix[point_index, j] = compute_g2(distance_vector2,
                                               g2_func.eta,
                                               g2_func.rcutoff,
                                               g2_func.rshift) * scaling
    end

    # Update affected atoms
    @inbounds for i in 1:(system_params.n_atoms)
        i == point_index && continue

        for (j, g2_func) in enumerate(nn_params.g2_functions)
            r_cutoff = g2_func.rcutoff
            dist1, dist2 = distance_vector1[i], distance_vector2[i]

            if (zero(T) < dist1 < r_cutoff) || (zero(T) < dist2 < r_cutoff)
                δg2 = compute_g2_element(dist2, g2_func.eta, r_cutoff, g2_func.rshift) -
                      compute_g2_element(dist1, g2_func.eta, r_cutoff, g2_func.rshift)
                g2_matrix[i, j] += δg2 * scaling
            end
        end
    end

    return g2_matrix
end

function compute_cos_angle(coordinates::AbstractMatrix{T},
                           box::AbstractVector{T},
                           i::Integer,
                           j::Integer,
                           k::Integer,
                           distance_ij::T,
                           distance_ik::T)::T where {T <: AbstractFloat}
    @assert i != j&&i != k && k != j "Indices must be different"

    atom_i = @view coordinates[:, i]
    vector_ij = compute_directional_vector(atom_i, @view(coordinates[:, j]), box)
    vector_ik = compute_directional_vector(atom_i, @view(coordinates[:, k]), box)

    cos_angle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)

    # -1 ≤ cos_angle ≤ 1
    cos_angle = clamp(cos_angle, -one(T), one(T))

    return cos_angle
end

function compute_triplet_geometry(coordinates::AbstractMatrix{T},
                                  box::AbstractVector{T},
                                  i::Integer,
                                  j::Integer,
                                  k::Integer,
                                  distance_ij::T,
                                  distance_ik::T)::Tuple{T, T} where {T <: AbstractFloat}
    @assert i != j&&i != k && k != j "Indices must be different"

    @inbounds begin
        atom_i = @view coordinates[:, i]
        atom_j = @view coordinates[:, j]
        atom_k = @view coordinates[:, k]

        distance_kj = compute_distance(atom_k, atom_j, box)
        vector_ij = compute_directional_vector(atom_i, atom_j, box)
        vector_ik = compute_directional_vector(atom_i, atom_k, box)

        cos_angle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)
        cos_angle = clamp(cos_angle, -one(T), one(T))
    end

    return cos_angle, distance_kj
end

function compute_g3_element(cos_angle::T,
                            distance_ij::T,
                            distance_ik::T,
                            distance_kj::T,
                            r_cutoff::T,
                            η::T,
                            ζ::T,
                            λ::T,
                            r_shift::T)::T where {T <: AbstractFloat}
    @fastmath begin
        angle_term = (one(T) + λ * cos_angle)^ζ

        squared_distances = (distance_ij - r_shift)^2 +
                            (distance_ik - r_shift)^2 +
                            (distance_kj - r_shift)^2
        radial_term = exp(-η * squared_distances)

        cutoff_term = distance_cutoff(distance_ij, r_cutoff) *
                      distance_cutoff(distance_ik, r_cutoff) *
                      distance_cutoff(distance_kj, r_cutoff)

        return angle_term * radial_term * cutoff_term
    end
end

function compute_g3(atom_index::Integer,
                    coordinates::AbstractMatrix{T},
                    box::AbstractVector{T},
                    distance_vector::AbstractVector{T},
                    r_cutoff::T,
                    η::T,
                    ζ::T,
                    λ::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    accumulator = zero(T)
    norm_factor = T(2)^(one(T) - ζ)

    @inbounds for k in eachindex(distance_vector)
        distance_ik = distance_vector[k]
        distance_ik <= zero(T) && continue

        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distance_vector[j]

            if zero(T) < distance_ij < r_cutoff && distance_ik < r_cutoff
                cos_angle, distance_kj = compute_triplet_geometry(coordinates, box, atom_index, j, k, distance_ij,
                                                                  distance_ik)

                contribution = compute_g3_element(cos_angle, distance_ij, distance_ik, distance_kj,
                                                  r_cutoff, η, ζ, λ, r_shift)

                accumulator += contribution
            end
        end
    end

    return norm_factor * accumulator
end

function build_g3_matrix(distance_matrix::AbstractMatrix{T},
                         coordinates::AbstractMatrix{T},
                         box::AbstractVector{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g3_functions = length(nn_params.g3_functions)
    g3_matrix = Matrix{T}(undef, n_atoms, n_g3_functions)

    @inbounds for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]

        for (j, g3_func) in enumerate(nn_params.g3_functions)
            g3_matrix[i, j] = compute_g3(i, coordinates, box, distance_vector,
                                         g3_func.rcutoff, g3_func.eta,
                                         g3_func.zeta, g3_func.lambda,
                                         g3_func.rshift)
        end
    end

    scaling = nn_params.symm_function_scaling
    return scaling == one(T) ? g3_matrix : rmul!(g3_matrix, scaling)
end

function update_g3_matrix!(g3_matrix::Matrix{T},
                           coordinates1::Matrix{T},
                           coordinates2::Matrix{T},
                           box::Vector{T},
                           distance_vec_1::Vector{T},
                           distance_vec_2::Vector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           displaced_atom_index::Integer)::Matrix{T} where {T <: AbstractFloat}
    scaling_factor::T = nn_params.symm_function_scaling
    n_atoms::Int = system_params.n_atoms

    @inbounds for central_atom_idx in 1:n_atoms
        if central_atom_idx == displaced_atom_index
            # Update matrix for displaced atom
            @inbounds for (g3_idx, g3_func) in enumerate(nn_params.g3_functions)
                g3_matrix[central_atom_idx, g3_idx] = compute_g3(displaced_atom_index,
                                                                 coordinates2,
                                                                 box,
                                                                 distance_vec_2,
                                                                 g3_func.rcutoff,
                                                                 g3_func.eta,
                                                                 g3_func.zeta,
                                                                 g3_func.lambda,
                                                                 g3_func.rshift) * scaling_factor
            end
        else
            # Update matrix for other atoms
            @inbounds for (g3_idx, g3_func) in enumerate(nn_params.g3_functions)
                r_cutoff::T = g3_func.rcutoff
                dist_ij_1::T = distance_vec_1[central_atom_idx]
                dist_ij_2::T = distance_vec_2[central_atom_idx]

                # Check if atom is within cutoff distance
                if (zero(T) < dist_ij_1 < r_cutoff) || (zero(T) < dist_ij_2 < r_cutoff)
                    central_atom_pos::Vector{T} = @view coordinates2[:, central_atom_idx]
                    delta_g3::T = zero(T)

                    # Calculate G3 changes for all third atoms
                    @inbounds for third_atom_idx in 1:n_atoms
                        # Skip if atoms are identical
                        if third_atom_idx == displaced_atom_index ||
                           third_atom_idx == central_atom_idx
                            continue
                        end

                        third_atom_pos::Vector{T} = @view coordinates2[:, third_atom_idx]
                        dist_ik::T = compute_distance(central_atom_pos, third_atom_pos, box)

                        # Check if third atom is within cutoff
                        if zero(T) < dist_ik < r_cutoff
                            displaced_pos_1::Vector{T} = @view coordinates1[:, displaced_atom_index]
                            displaced_pos_2::Vector{T} = @view coordinates2[:, displaced_atom_index]

                            dist_kj_1::T = compute_distance(displaced_pos_1, third_atom_pos, box)
                            dist_kj_2::T = compute_distance(displaced_pos_2, third_atom_pos, box)

                            if (zero(T) < dist_kj_1 < r_cutoff) ||
                               (zero(T) < dist_kj_2 < r_cutoff)
                                # Calculate angle vectors
                                vec_ij_1::Vector{T} = compute_directional_vector(central_atom_pos, displaced_pos_1, box)
                                vec_ij_2::Vector{T} = compute_directional_vector(central_atom_pos, displaced_pos_2, box)
                                vec_ik::Vector{T} = compute_directional_vector(central_atom_pos, third_atom_pos, box)

                                # Calculate cosine angles
                                cos_angle_1::T = dot(vec_ij_1, vec_ik) / (dist_ij_1 * dist_ik)
                                cos_angle_2::T = dot(vec_ij_2, vec_ik) / (dist_ij_2 * dist_ik)

                                # Ensure angles are valid
                                @assert -one(T) <= cos_angle_1 <= one(T)
                                @assert -one(T) <= cos_angle_2 <= one(T)

                                # Calculate G3 differences
                                g3_val_1::T = compute_g3_element(cos_angle_1, dist_ij_1, dist_ik, dist_kj_1,
                                                                 r_cutoff, g3_func.eta, g3_func.zeta,
                                                                 g3_func.lambda, g3_func.rshift)
                                g3_val_2::T = compute_g3_element(cos_angle_2, dist_ij_2, dist_ik, dist_kj_2,
                                                                 r_cutoff, g3_func.eta, g3_func.zeta,
                                                                 g3_func.lambda, g3_func.rshift)

                                delta_g3 += T(2)^(one(T) - g3_func.zeta) * (g3_val_2 - g3_val_1)
                            end
                        end
                    end

                    # Update matrix element with accumulated changes
                    g3_matrix[central_atom_idx, g3_idx] += delta_g3 * scaling_factor
                end
            end
        end
    end

    return g3_matrix
end

function compute_g9_element(cos_angle::T,
                            distance_ij::T,
                            distance_ik::T,
                            r_cutoff::T,
                            η::T,
                            ζ::T,
                            λ::T,
                            r_shift::T)::T where {T <: AbstractFloat}
    @fastmath begin
        angle_term = (one(T) + λ * cos_angle)^ζ

        r_ij = distance_ij - r_shift
        r_ik = distance_ik - r_shift
        radial_term = exp(-η * (r_ij^2 + r_ik^2))

        cutoff_term = distance_cutoff(distance_ij, r_cutoff) *
                      distance_cutoff(distance_ik, r_cutoff)

        return angle_term * radial_term * cutoff_term
    end
end

function compute_g9(atom_index::Integer,
                    coordinates::Matrix{T},
                    box::Vector{T},
                    distance_vector::Vector{T},
                    r_cutoff::T,
                    η::T,
                    ζ::T,
                    λ::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    accumulator = zero(T)

    @inbounds for k in eachindex(distance_vector)
        distance_ik = distance_vector[k]

        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distance_vector[j]

            if zero(T) < distance_ij < r_cutoff && zero(T) < distance_ik < r_cutoff
                cos_angle = compute_cos_angle(coordinates,
                                              box,
                                              atom_index,
                                              j,
                                              k,
                                              distance_ij,
                                              distance_ik)

                accumulator += compute_g9_element(cos_angle,
                                                  distance_ij,
                                                  distance_ik,
                                                  r_cutoff,
                                                  η,
                                                  ζ,
                                                  λ,
                                                  r_shift)
            end
        end
    end

    return (T(2)^(one(T) - ζ) * accumulator)
end

function build_g9_matrix(distance_matrix::AbstractMatrix{T},
                         coordinates::AbstractMatrix{T},
                         box::AbstractVector{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g9_functions = length(nn_params.g9_functions)
    g9_matrix = Matrix{T}(undef, n_atoms, n_g9_functions)

    @inbounds for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]

        for (j, g9_func) in enumerate(nn_params.g9_functions)
            g9_matrix[i, j] = compute_g9(i, coordinates, box, distance_vector,
                                         g9_func.rcutoff, g9_func.eta,
                                         g9_func.zeta, g9_func.lambda,
                                         g9_func.rshift)
        end
    end

    scaling = nn_params.symm_function_scaling
    return scaling == one(T) ? g9_matrix : rmul!(g9_matrix, scaling)
end

function update_g9_matrix!(g9_matrix::AbstractMatrix{T},
                           coordinates1::AbstractMatrix{T},
                           coordinates2::AbstractMatrix{T},
                           box::AbstractVector{T},
                           distance_vector1::AbstractVector{T},
                           distance_vector2::AbstractVector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           displaced_atom_index::Integer) where {T <: AbstractFloat}
    for selected_atom_index in 1:(system_params.n_atoms)
        if selected_atom_index == displaced_atom_index
            for (g9_index, g9_func) in enumerate(nn_params.g9_functions)
                g9_matrix[selected_atom_index, g9_index] = compute_g9(displaced_atom_index, coordinates2, box,
                                                                      distance_vector2,
                                                                      g9_func.rcutoff, g9_func.eta, g9_func.zeta,
                                                                      g9_func.lambda, g9_func.rshift) *
                                                           nn_params.symm_function_scaling
            end
        else
            for (g9_index, g9_func) in enumerate(nn_params.g9_functions)
                distance_ij_1 = distance_vector1[selected_atom_index]
                distance_ij_2 = distance_vector2[selected_atom_index]

                if 0 < distance_ij_2 < g9_func.rcutoff || 0 < distance_ij_1 < g9_func.rcutoff
                    Δg9 = zero(T)

                    for third_atom_index in 1:(system_params.n_atoms)
                        if third_atom_index != displaced_atom_index && third_atom_index != selected_atom_index
                            selected_atom = @view coordinates2[:, selected_atom_index]
                            third_atom = @view coordinates2[:, third_atom_index]
                            distance_ik = compute_distance(selected_atom, third_atom, box)

                            if 0 < distance_ik < g9_func.rcutoff
                                displaced_atom_1 = @view coordinates1[:, displaced_atom_index]
                                displaced_atom_2 = @view coordinates2[:, displaced_atom_index]

                                vector_ij_1 = compute_directional_vector(selected_atom, displaced_atom_1, box)
                                vector_ij_2 = compute_directional_vector(selected_atom, displaced_atom_2, box)
                                vector_ik = compute_directional_vector(selected_atom, third_atom, box)

                                cos_angle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                cos_angle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)

                                @assert -1≤cos_angle1≤1 "Invalid cosine value: $cos_angle1"
                                @assert -1≤cos_angle2≤1 "Invalid cosine value: $cos_angle2"

                                g9_1 = compute_g9_element(cos_angle1, distance_ij_1, distance_ik,
                                                          g9_func.rcutoff, g9_func.eta, g9_func.zeta,
                                                          g9_func.lambda, g9_func.rshift)
                                g9_2 = compute_g9_element(cos_angle2, distance_ij_2, distance_ik,
                                                          g9_func.rcutoff, g9_func.eta, g9_func.zeta,
                                                          g9_func.lambda, g9_func.rshift)

                                Δg9 += 2^(1 - g9_func.zeta) * (g9_2 - g9_1)
                            end
                        end
                    end

                    g9_matrix[selected_atom_index, g9_index] += Δg9 * nn_params.symm_function_scaling
                end
            end
        end
    end

    return g9_matrix
end
