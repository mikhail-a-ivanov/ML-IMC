"""
function computeG2Element(R, Rc, Rs, η)

Computes a single exponent
of the G2 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG2Element(distance, eta, rcutoff, rshift)
    return (exp(-eta * (distance - rshift)^2) * distanceCutoff(distance, rcutoff))
end

"""
function computeG2(distances, rcutoff, rshift, eta)

Computes the total G2 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG2(distances, eta, rcutoff, rshift)
    sum = 0
    @simd for distance in distances
        sum += computeG2Element(distance, eta, rcutoff, rshift)
    end
    return (sum)
end

"""
function buildG2Matrix(distanceMatrix, G2Functions)

Builds a matrix of G2 values with varying Rs and η parameters for each atom in the configuration
"""
function buildG2Matrix(distanceMatrix, G2Functions::Vector{G2})
    N = size(distanceMatrix)[1]
    G2Matrix = Matrix{Float64}(undef, N, length(G2Functions))
    for i = 1:N
        distanceVector = distanceMatrix[i, :]
        for (j, G2Function) in enumerate(G2Functions)
            eta = G2Function.eta
            rcutoff = G2Function.rcutoff
            rshift = G2Function.rshift
            G2Matrix[i, j] = computeG2(distanceVector, eta, rcutoff, rshift)
        end
    end
    return (G2Matrix)
end

"""
function updateG2Matrix!(G2Matrix, distanceVector1, distanceVector2, systemParms, G2Functions, pointIndex)

Updates the G2 matrix with the displacement of a single atom
"""
function updateG2Matrix!(
    G2Matrix,
    distanceVector1,
    distanceVector2,
    systemParms,
    G2Functions::Vector{G2},
    pointIndex,
)
    for i = 1:systemParms.N
        # Rebuild the whole G2 matrix column for the displaced particle
        if i == pointIndex
            for (j, G2Function) in enumerate(G2Functions)
                eta = G2Function.eta
                rcutoff = G2Function.rcutoff
                rshift = G2Function.rshift
                G2Matrix[pointIndex, j] =
                    computeG2(distanceVector2, eta, rcutoff, rshift)
            end
            # Compute the change in G2 caused by the displacement of an atom
        else
            for (j, G2Function) in enumerate(G2Functions)
                rcutoff = G2Function.rcutoff
                if 0.0 < distanceVector2[i] < rcutoff || 0.0 < distanceVector1[i] < rcutoff
                    eta = G2Function.eta
                    rshift = G2Function.rshift
                    G2_1 = computeG2Element(distanceVector1[i], eta, rcutoff, rshift)
                    G2_2 = computeG2Element(distanceVector2[i], eta, rcutoff, rshift)
                    ΔG2 = G2_2 - G2_1
                    G2Matrix[i, j] += ΔG2
                end
            end
        end
    end
    return (G2Matrix)
end