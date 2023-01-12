"""
function computePMF(refRDF, systemParms)

Compute PMF for a given system (in kT units)
"""
function computePMF(refRDF, systemParms)
    PMF = zeros(Float64, systemParms.Nbins)
    repulsionRegion = refRDF .== 0
    repulsionPoints = length(repulsionRegion[repulsionRegion .!= 0])
    maxPMFIndex = repulsionPoints + 1
    maxPMF = -log(refRDF[maxPMFIndex]) / systemParms.β
    secondMaxPMF = -log(refRDF[maxPMFIndex + 1]) / systemParms.β
    diffPMF = maxPMF - secondMaxPMF

    for i in eachindex(PMF)
        if repulsionRegion[i]
            PMF[i] = maxPMF + diffPMF * (maxPMFIndex - i)
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

"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, NNParms)

Computes loss gradients for one frame
"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, NNParms)
    parameters = Flux.params(model)
    loss = (ΔENN - ΔEPMF)^2
    regloss = sum(parameters[1] .^ 2) * NNParms.REGP
    println("   Energy loss: $(round(loss, digits=4))")
    println("   PMF energy difference: $(round(ΔEPMF, digits=4))")
    println("   NN energy difference: $(round(ΔENN, digits=4))")
    println("   Regularization loss: $(round(regloss, digits=4))")

    # Compute dL/dw
    ENN1Gradients = computeEnergyGradients(G2Matrix1, model)
    ENN2Gradients = computeEnergyGradients(G2Matrix2, model)
    gradientScaling = 2 * (ΔENN - ΔEPMF)

    lossGradient = gradientScaling .* (ENN2Gradients .- ENN1Gradients)
    regLossGradient = @. parameters * 2 * NNParms.REGP
    lossGradient += regLossGradient
    return (lossGradient)
end


function pretrainingMCMove!(frameInputArrays, PMF, model, NNParms, systemParms, rng)
    # Unpack mcarrays
    frame, distanceMatrix, G2Matrix1 = frameInputArrays

    # Compute energy of the initial configuration
    ENN1Vector = totalEnergyVectorInit(G2Matrix1, model)
    ENN1 = sum(ENN1Vector)
    EPMF1 = computePMFEnergy(PMF, distanceMatrix, systemParms)

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(systemParms.N))

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Displace the particle
    dr = [
        systemParms.Δ * (rand(rng, Float64) - 0.5),
        systemParms.Δ * (rand(rng, Float64) - 0.5),
        systemParms.Δ * (rand(rng, Float64) - 0.5),
    ]

    positions(frame)[:, pointIndex] .+= dr

    # Compute the updated distance vector
    distanceVector2 = Array{Float64}(undef, systemParms.N)
    distanceVector2 = updateDistance!(frame, distanceVector2, pointIndex)

    indexesForUpdate = getBoolMaskForUpdating(distanceVector2, systemParms, NNParms)

    # Update the distance matrix
    distanceMatrix[pointIndex, :] = distanceVector2
    distanceMatrix[:, pointIndex] = distanceVector2

    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(G2Matrix1)
    updateG2Matrix!(
        G2Matrix2,
        distanceVector1,
        distanceVector2,
        systemParms,
        NNParms,
        pointIndex,
    )

    # Compute the energy again
    ENN2Vector = totalEnergyVector(G2Matrix2, model, indexesForUpdate, ENN1Vector)
    ENN2 = sum(ENN2Vector)
    EPMF2 = computePMFEnergy(PMF, distanceMatrix, systemParms)

    # Get energy differences
    ΔENN = ENN2 - ENN1
    ΔEPMF = EPMF2 - EPMF1

    # Revert the changes
    positions(frame)[:, pointIndex] .-= dr
    distanceMatrix[pointIndex, :] = distanceVector1
    distanceMatrix[:, pointIndex] = distanceVector1

    # Pack mcarrays
    frameOutputArrays = (frame, distanceMatrix, G2Matrix1, G2Matrix2)

    return (frameOutputArrays, ΔENN, ΔEPMF)
end

function preTrainMC!(NNParms, systemParmsList, model, opt, refRDFs, steps=100)
    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    println("Running $(steps) steps of pre-training Monte-Carlo...\n")
    nsystems = length(systemParmsList)

    # Compute PMF for each system and read trajectories
    PMFs = []
    trajectories = []
    for systemId = 1:nsystems
        PMF = computePMF(refRDFs[systemId], systemParmsList[systemId])
        append!(PMFs, [PMF])
        traj = readXTC(systemParmsList[systemId])
        append!(trajectories, [traj])
    end

    for step = 1:steps
        println("\nStep $(step)...\n")
        lossGradients = []
        for systemId = 1:nsystems
            # Read a frame from the trajectory
            nframes = Int(size(trajectories[systemId])) - 1
            frameId = rand(rngXor, 1:nframes) # Don't take the first frame
            frame = read_step(trajectories[systemId], frameId)

            # Build distance and G2 matrices
            distanceMatrix = buildDistanceMatrix(frame)
            G2Matrix = buildG2Matrix(distanceMatrix, NNParms)
            frameInputArrays = (frame, distanceMatrix, G2Matrix)

            # Run an MC step, compute energy differences with PMF and the neural network,
            # restore all input arguments to their original state
            frameOutputArrays, ΔENN, ΔEPMF = pretrainingMCMove!(
                frameInputArrays, PMFs[systemId], model, NNParms, systemParmsList[systemId], rngXor)
            frame, distanceMatrix, G2Matrix1, G2Matrix2 = frameOutputArrays
            # Compute the loss gradient
            lossGradient = computePreTrainingLossGradients(
                ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, NNParms)
            append!(lossGradients, [lossGradient])
        end
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        updatemodel!(model, opt, meanLossGradients)
    end
    @save "model-pre-trained.bson" model
    return (model)
end