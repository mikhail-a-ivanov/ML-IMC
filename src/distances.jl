"""
function builddistanceMatrix(frame)

Builds the distance matrix for a given
Chemfiles frame

Note that the Chemfiles distance
function starts indexing atoms from 0!
"""
function buildDistanceMatrix(frame)
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
distanceCutoff(R, Rc = 10)

Cutoff distance function (J. Chem. Phys. 134, 074106 (2011))
"""
function distanceCutoff(R, Rc = 10)
    if R > Rc
        return (0.0)
    else
        return (0.5 * (cos(π * R / Rc) + 1))
    end
end
