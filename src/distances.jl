using ..ML_IMC

"""
Build distance matrix for all atoms in the frame.
"""
function build_distance_matrix(frame::Frame)::Matrix{Float64}
    coordinates = positions(frame)
    n_atoms = length(frame)
    box = lengths(UnitCell(frame))

    distance_matrix = Matrix{Float64}(undef, n_atoms, n_atoms)

    @inbounds for i in 1:n_atoms
        distance_matrix[i, i] = 0.0
        @simd for j in (i + 1):n_atoms
            dist = compute_distance(coordinates[:, i], coordinates[:, j], box)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        end
    end

    return distance_matrix
end

"""
Compute distance between two points considering periodic boundary conditions.
"""
function compute_distance(r1::AbstractVector{T}, r2::AbstractVector{T},
                          box::AbstractVector{T})::T where {T <: AbstractFloat}
    @fastmath return sqrt(sum(compute_squared_distance_component(r1[i], r2[i], box[i]) for i in eachindex(r1, r2, box)))
end

"""
Compute distances from one point to all other points.
"""
function compute_distance_vector(r1::AbstractVector{T}, coordinates::AbstractMatrix{T},
                                 box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    @fastmath return [sqrt(sum(compute_squared_distance_component(r1[i], coordinates[i, j], box[i])
                               for i in axes(coordinates, 1)))
                      for j in axes(coordinates, 2)]
end

"""
Compute squared distance component along one dimension with periodic boundary conditions.
"""
@inline function compute_squared_distance_component(x1::T, x2::T, box_size::T)::T where {T <: AbstractFloat}
    dx = x2 - x1
    dx -= box_size * round(dx / box_size)
    return dx * dx
end

"""
Compute directional vector between two points considering periodic boundary conditions.
"""
function compute_directional_vector(r1::AbstractVector{T}, r2::AbstractVector{T},
                                    box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    return [compute_distance_component(r1[i], r2[i], box[i]) for i in eachindex(r1, r2, box)]
end

"""
Compute distance component along one dimension with periodic boundary conditions.
"""
@inline function compute_distance_component(x1::T, x2::T, box_size::T)::T where {T <: AbstractFloat}
    dx = x2 - x1
    dx -= box_size * round(dx / box_size)
    return dx
end

# ---------------------------------------------------------------------------------------
# ---------- Chemfiles Functions

"""
Build distance matrix using Chemfiles native distance calculation.
"""
function build_distance_matrix_chemfiles(frame::Frame)::Matrix{Float64}
    n_atoms = length(frame)
    distance_matrix = Matrix{Float64}(undef, n_atoms, n_atoms)

    @inbounds for i in 1:n_atoms
        distance_matrix[i, i] = 0.0
        @simd for j in (i + 1):n_atoms
            # Note: Chemfiles uses 0-based indexing
            dist = Chemfiles.distance(frame, i - 1, j - 1)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        end
    end

    return distance_matrix
end

"""
Update distance vector using Chemfiles native distance calculation.
"""
function update_distance_vector_chemfiles!(frame::Frame, distance_vector::Vector{Float64},
                                           point_index::Int)::Vector{Float64}
    @inbounds @simd for i in eachindex(distance_vector)
        # Note: Chemfiles uses 0-based indexing
        distance_vector[i] = Chemfiles.distance(frame, i - 1, point_index - 1)
    end
    return distance_vector
end
