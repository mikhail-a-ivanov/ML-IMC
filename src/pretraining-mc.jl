"""
function computePMF(refRDF, systemParms)

Compute PMF for a given system (in kT units)
"""
function computePMF(refRDF, systemParms, potentialWall=1000)
    PMF = zeros(Float64, systemParms.Nbins)
    repulsionRegion = refRDF .== 0
    for i in eachindex(PMF)
        if repulsionRegion[i]
            PMF[i] = potentialWall
        else
            PMF[i] = -log(refRDF[i]) / systemParms.β
        end
    end
    return (PMF)
end

"""
function computePMFEnergy(PMF, distanceMatrix, systemParms)

Compute a total energy of a configuration with PMF
"""
function computePMFEnergy(PMF, distanceMatrix, systemParms)
    hist = zeros(Float64, systemParms.Nbins)
    hist = hist!(distanceMatrix, hist, systemParms)
    E = sum(hist .* PMF)
    return (E)
end
