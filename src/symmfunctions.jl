"""
function computeG2Element(distance, eta, rcutoff, rshift)::Float64

Computes a single exponent
of the G2 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG2Element(distance, eta, rcutoff, rshift)::Float64
    return (exp(-eta * (distance - rshift)^2) * distanceCutoff(distance, rcutoff))
end

"""
function computeG2(distances, eta, rcutoff, rshift)::Float64

Computes the total G2 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG2(distances, eta, rcutoff, rshift)::Float64
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

"""
function computeCosAngle(coordinates, box, i, j, k, distance_ij, distance_ik)::Float64

Computes cosine of angle in a triplet of atoms ijk with atom i as the central atom
"""
function computeCosAngle(coordinates, box, i, j, k, distance_ij, distance_ik)::Float64
    @assert i != j && i != k && k != j
    vector_0i = coordinates[:, i]
    vector_ij = computeDirectionalVector(vector_0i, coordinates[:, j], box)
    vector_ik = computeDirectionalVector(vector_0i, coordinates[:, k], box)
    cosAngle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)
    @assert -1.0 <= cosAngle <= 1.0
    return (cosAngle)
end

"""
function computeTripletGeometry(coordinates, box, i, j, k, distance_ij, distance_ik)::Tuple{Float64, Float64}

Computes cosine of angle in a triplet of atoms ijk with atom i as the central atom
and the distance kj for G3 symmetry function
"""
function computeTripletGeometry(coordinates, box, i, j, k, distance_ij, distance_ik)::Tuple{Float64,Float64}
    @assert i != j && i != k && k != j
    vector_0i = coordinates[:, i]
    vector_0j = coordinates[:, j]
    vector_0k = coordinates[:, k]
    distance_kj = computeDistance(vector_0k, vector_0j, box)
    vector_ij = computeDirectionalVector(vector_0i, vector_0j, box)
    vector_ik = computeDirectionalVector(vector_0i, vector_0k, box)
    cosAngle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)
    @assert -1.0 <= cosAngle <= 1.0
    return (cosAngle, distance_kj)
end

"""
function computeG3element(cosAngle, distance_ij, distance_ik, rcutoff, eta, zeta, lambda, rshift)::Float64

Computes a single exponent
of the G3 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG3element(cosAngle, distance_ij, distance_ik, distance_kj, rcutoff, eta, zeta, lambda, rshift)::Float64
    return (
        (1.0 + lambda * cosAngle)^zeta *
        exp(-eta * (
            (distance_ij - rshift)^2 +
            (distance_ik - rshift)^2 +
            (distance_kj - rshift)^2)) *
        distanceCutoff(distance_ij, rcutoff) *
        distanceCutoff(distance_ik, rcutoff) *
        distanceCutoff(distance_kj, rcutoff))
end

"""
function computeG3(i, coordinates, box, distanceVector, cutoff, eta, zeta, lambda, rshift)::Float64

Computes the total G3 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG3(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)::Float64
    sum = 0.0
    @inbounds for k in eachindex(distanceVector)
        distance_ik = distanceVector[k]
        @inbounds @simd for j in 1:k-1
            distance_ij = distanceVector[j]
            if 0 < distance_ij < rcutoff && 0 < distance_ik < rcutoff
                cosAngle, distance_kj = computeTripletGeometry(coordinates, box, i, j, k, distance_ij, distance_ik)
                sum += computeG3element(cosAngle, distance_ij, distance_ik, distance_kj, rcutoff, eta, zeta, lambda, rshift)
            end
        end
    end
    return (2.0^(1.0 - zeta) * sum)
end

"""
function buildG3Matrix(distanceMatrix, coordinates, box, G3Functions)

Builds a matrix of G3 values
"""
function buildG3Matrix(distanceMatrix, coordinates, box, G3Functions::Vector{G3})
    N = size(distanceMatrix)[1]
    G3Matrix = Matrix{Float64}(undef, N, length(G3Functions))
    for i = 1:N
        distanceVector = distanceMatrix[i, :]
        for (j, G3Function) in enumerate(G3Functions)
            eta = G3Function.eta
            lambda = G3Function.lambda
            zeta = G3Function.zeta
            rcutoff = G3Function.rcutoff
            rshift = G3Function.rshift
            G3Matrix[i, j] = computeG3(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)
        end
    end
    return (G3Matrix)
end

"""
function updateG3Matrix!(G3Matrix, coordinates1, coordinates2, box,
    distanceVector1, distanceVector2, systemParms, G3Functions, displacedAtomIndex)

Updates the G3 matrix with the displacement of a single atom
"""
function updateG3Matrix!(
    G3Matrix,
    coordinates1,
    coordinates2,
    box,
    distanceVector1,
    distanceVector2,
    systemParms,
    G3Functions::Vector{G3},
    displacedAtomIndex,
)
    for selectedAtomIndex = 1:systemParms.N
        # Rebuild the whole G3 matrix column for the displaced atom
        if selectedAtomIndex == displacedAtomIndex
            for (G3Index, G3Function) in enumerate(G3Functions)
                eta = G3Function.eta
                lambda = G3Function.lambda
                zeta = G3Function.zeta
                rcutoff = G3Function.rcutoff
                rshift = G3Function.rshift
                G3Matrix[selectedAtomIndex, G3Index] = computeG3(displacedAtomIndex, coordinates2, box, 
                        distanceVector2, rcutoff, eta, zeta, lambda, rshift)
            end
            # Compute the change in G3 caused by the displacement of an atom
            # New ijk triplet description
            # Central atom (i): selectedAtomIndex
            # Second atom (j): displacedAtomIndex
            # Third atom (k): thirdAtomIndex
        else
            for (G3Index, G3Function) in enumerate(G3Functions)
                rcutoff = G3Function.rcutoff
                distance_ij_1 = distanceVector1[selectedAtomIndex]
                distance_ij_2 = distanceVector2[selectedAtomIndex]
                # Ignore if the selected atom is far from the displaced atom
                if 0.0 < distance_ij_2 < rcutoff || 0.0 < distance_ij_1 < rcutoff
                    eta = G3Function.eta
                    lambda = G3Function.lambda
                    zeta = G3Function.zeta
                    rshift = G3Function.rshift
                    # Accumulate differences for the selected atom
                    # over all third atoms
                    ΔG3 = 0.0
                    # Loop over all ik pairs
                    for thirdAtomIndex in 1:systemParms.N
                        # Make sure i != j != k
                        if thirdAtomIndex != displacedAtomIndex && thirdAtomIndex != selectedAtomIndex
                            # It does not make a difference whether
                            # coordinates2 or coordinates1 are used -
                            # both selectedAtom and thirdAtom have
                            # have the same coordinates in the old and 
                            # the updated configuration
                            selectedAtom = coordinates2[:, selectedAtomIndex]
                            thirdAtom = coordinates2[:, thirdAtomIndex]
                            distance_ik = computeDistance(selectedAtom, thirdAtom, box)
                            # The current ik pair is fixed so if r_ik > rcutoff
                            # no change in this G3(i,j,k) is needed
                            if 0.0 < distance_ik < rcutoff
                                # Compute the contribution to the change
                                # from the old configuration
                                displacedAtom_1 = coordinates1[:, displacedAtomIndex]
                                displacedAtom_2 = coordinates2[:, displacedAtomIndex]
                                distance_kj_1 = computeDistance(displacedAtom_1, thirdAtom, box)
                                distance_kj_2 = computeDistance(displacedAtom_2, thirdAtom, box)
                                if 0.0 < distance_kj_1 < rcutoff || 0.0 < distance_kj_2 < rcutoff
                                    # Compute cos of angle
                                    vector_ij_1 = computeDirectionalVector(selectedAtom, displacedAtom_1, box)
                                    vector_ij_2 = computeDirectionalVector(selectedAtom, displacedAtom_2, box)
                                    vector_ik = computeDirectionalVector(selectedAtom, thirdAtom, box)
                                    cosAngle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                    cosAngle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)
                                    @assert -1.0 <= cosAngle1 <= 1.0
                                    @assert -1.0 <= cosAngle2 <= 1.0
                                    # Compute differences in G3
                                    G3_1 = computeG3element(cosAngle1, distance_ij_1, distance_ik, distance_kj_1, 
                                        rcutoff, eta, zeta, lambda, rshift)
                                    G3_2 = computeG3element(cosAngle2, distance_ij_2, distance_ik, distance_kj_2, 
                                        rcutoff, eta, zeta, lambda, rshift)
                                    ΔG3 += 2.0^(1.0 - zeta) * (G3_2 - G3_1)
                                end
                            end
                        end
                    end
                    # Apply the computed differences
                    G3Matrix[selectedAtomIndex, G3Index] += ΔG3
                end
            end
        end
    end
    return (G3Matrix)
end


"""
function computeG9element(cosAngle, distance_ij, distance_ik, rcutoff, eta, zeta, lambda, rshift)::Float64

Computes a single exponent
of the G9 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG9element(cosAngle, distance_ij, distance_ik, rcutoff, eta, zeta, lambda, rshift)::Float64
    return (
        (1.0 + lambda * cosAngle)^zeta *
        exp(-eta * (
            (distance_ij - rshift)^2 +
            (distance_ik - rshift)^2)
        ) *
        distanceCutoff(distance_ij, rcutoff) *
        distanceCutoff(distance_ik, rcutoff))
end


"""
function computeG9(i, coordinates, box, distanceVector, cutoff, eta, zeta, lambda, rshift)::Float64

Computes the total G9 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function computeG9(i, coordinates, box, distanceVector, cutoff, eta, zeta, lambda, rshift)::Float64
    sum = 0.0
    @inbounds for k in eachindex(distanceVector)
        distance_ik = distanceVector[k]
        @inbounds @simd for j in 1:k-1
            distance_ij = distanceVector[j]
            if 0 < distance_ij < cutoff && 0 < distance_ik < cutoff
                cosAngle = computeCosAngle(coordinates, box, i, j, k, distance_ij, distance_ik)
                sum += computeG9element(cosAngle, distance_ij, distance_ik, cutoff, eta, zeta, lambda, rshift)
            end
        end
    end
    return (2.0^(1.0 - zeta) * sum)
end

"""
function buildG9Matrix(distanceMatrix, coordinates, box, G9Functions)

Builds a matrix of G9 values
"""
function buildG9Matrix(distanceMatrix, coordinates, box, G9Functions::Vector{G9})
    N = size(distanceMatrix)[1]
    G9Matrix = Matrix{Float64}(undef, N, length(G9Functions))
    for i = 1:N
        distanceVector = distanceMatrix[i, :]
        for (j, G9Function) in enumerate(G9Functions)
            eta = G9Function.eta
            lambda = G9Function.lambda
            zeta = G9Function.zeta
            rcutoff = G9Function.rcutoff
            rshift = G9Function.rshift
            G9Matrix[i, j] = computeG9(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)
        end
    end
    return (G9Matrix)
end