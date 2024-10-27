using Chemfiles

function build_distance_matrix_chemfiles(frame::Frame)::Matrix{Float64}
    n_atoms = length(frame)
    distance_matrix = Matrix{Float64}(undef, n_atoms, n_atoms)

    @inbounds for i in 1:n_atoms
        distance_matrix[i, i] = 0.0
        for j in (i + 1):n_atoms
            dist = distance(frame, i - 1, j - 1)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        end
    end

    return distance_matrix
end

function update_distance!(frame::Frame, distance_vector::Vector{Float64}, point_index::Int)::Vector{Float64}
    @inbounds for i in eachindex(distance_vector)
        distance_vector[i] = distance(frame, i - 1, point_index - 1) # NOTE: Chemfiles.distance 0-based index
    end
    return distance_vector
end

function distance_cutoff(distance::Float64, r_cutoff::Float64=6.0)::Float64
    if distance > r_cutoff
        return 0.0
    else
        return (0.5 * (cos(π * distance / r_cutoff) + 1.0))
    end
end

function compute_distance_component(x1::Float64, x2::Float64, box_size::Float64)::Float64
    dx = x2 - x1
    dx -= box_size * round(dx / box_size)
    return dx
end

function compute_squared_distance_component(x1::Float64, x2::Float64, box_size::Float64)::Float64
    dx = x2 - x1
    dx -= box_size * round(dx / box_size)
    return dx * dx
end

function compute_directional_vector(r1::AbstractVector{T}, r2::AbstractVector{T},
                                    box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    return [compute_distance_component(r1[i], r2[i], box[i]) for i in eachindex(r1, r2, box)]
end

function compute_distance(r1::AbstractVector{T}, r2::AbstractVector{T},
                          box::AbstractVector{T})::T where {T <: AbstractFloat}
    return sqrt(sum(compute_squared_distance_component(r1[i], r2[i], box[i]) for i in eachindex(r1, r2, box)))
end

function compute_distance_vector(r1::AbstractVector{T}, coordinates::AbstractMatrix{T},
                                 box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    return [sqrt(sum(compute_squared_distance_component(r1[i], coordinates[i, j], box[i]) for i in axes(coordinates, 1)))
            for j in axes(coordinates, 2)]
end

function build_distance_matrix(frame::Frame)::Matrix{Float64}
    coordinates = positions(frame)
    n_atoms = length(frame)
    box = lengths(UnitCell(frame))

    distance_matrix = Matrix{Float64}(undef, n_atoms, n_atoms)

    @inbounds for i in 1:n_atoms
        distance_matrix[i, i] = 0.0
        for j in (i + 1):n_atoms
            dist = compute_distance(coordinates[:, i], coordinates[:, j], box)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        end
    end

    return distance_matrix
end
