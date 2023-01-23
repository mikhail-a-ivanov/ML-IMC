"""
buildDistanceMatrixChemfiles(frame)

Builds the distance matrix for a given
Chemfiles frame

Note that the Chemfiles distance
function starts indexing atoms from 0!
"""
function buildDistanceMatrixChemfiles(frame)
    N = length(frame)
    distanceMatrix = Array{Float64}(undef, (N, N))
    @inbounds for i = 0:N-1
        @inbounds for j = 0:N-1
            distanceMatrix[i+1, j+1] = distance(frame, i, j)
        end
    end
    return (distanceMatrix)
end

"""
function updatedistance!(frame, distanceVector, pointIndex)

Updates all distances between a selected particle
and all the others in a given configuration

Note that the Chemfiles distance
function starts indexing atoms from 0!

pointIndex can be any number from 1 to N,
so I need to shift it by -1 so it takes
the same values as the iterator i
"""
function updateDistance!(frame, distanceVector, pointIndex)
    @fastmath @inbounds for i = 0:length(distanceVector)-1
        distanceVector[i+1] = distance(frame, i, pointIndex - 1)
    end
    return (distanceVector)
end

"""
function distanceCutoff(distance, rcutoff = 6.0)

Cutoff distance function (J. Chem. Phys. 134, 074106 (2011))
"""
function distanceCutoff(distance, rcutoff=6.0)
    if distance > rcutoff
        return (0.0)
    else
        return (0.5 * (cos(π * distance / rcutoff) + 1.0))
    end
end

"""
function squaredDistanceComponent(x1, x2, xsize)

Computes distance between two points along a selected axis,
taking the periodic boundary conditions into the account.
The result is then squared.
"""
function squaredDistanceComponent(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int32, round(dx / xsize))
    return dx^2
end

"""
function computeDistance(r1, r2, box)

Computes PBC distance between two points
"""
function computeDistance(r1, r2, box)
    return sqrt.(reduce(+, map(squaredDistanceComponent, r1, r2, box)))
end


"""
computeDistanceVector(r1, coordinates, box)

Computes a vector of PBC distances between point r1
and all the others in the simulation box
"""
function computeDistanceVector(r1, coordinates, box)
    return vec(sqrt.(sum(broadcast(squaredDistanceComponent, r1, coordinates, box), dims=1)))
end


"""
buildDistanceMatrix(frame)

Builds the distance matrix for a given
Chemfiles frame using my own distance functions
(slightly faster but more memory consumption)
"""
function buildDistanceMatrix(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

    distanceMatrix = Matrix{Float64}(undef, N, N)
    for i in 1:N
        distanceMatrix[i, :] = computeDistanceVector(coordinates[:, i], coordinates, box)
    end

    return (distanceMatrix)
end    