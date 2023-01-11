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

"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, NNParms)

Computes loss gradients for one frame
"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, NNParms)
    N = size(G2Matrix1)[1]
    parameters = Flux.params(model)
    loss = (ΔENN - ΔEPMF)^2 / N
    regloss = sum(parameters[1] .^ 2) * NNParms.REGP
    println("   Energy loss (per atom): $(round(loss, digits=4))")
    println("   PMF energy difference: $(round(ΔEPMF, digits=4))")
    println("   NN energy difference: $(round(ΔENN, digits=4))")
    println("   Regularization loss: $(round(regloss, digits=4))")

    # Compute dL/dw
    ENN1Gradients = computeEnergyGradients(G2Matrix1, model)
    ENN2Gradients = computeEnergyGradients(G2Matrix2, model)
    gradientScaling = 2 / N * (ΔENN - ΔEPMF)

    lossGradient = gradientScaling .* (ENN2Gradients .- ENN1Gradients)
    regLossGradient = @. parameters * 2 * NNParms.REGP
    lossGradient += regLossGradient
    return (lossGradient)
end


function pretrainingMCMove!(frameInputArrays, PMF, model, NNParms, systemParms, rng)
    # Unpack mcarrays
    frame, distanceMatrix, G2Matrix1 = frameInputArrays

    # Compute energy of the initial configuration
    ENN1 = totalEnergyScalar(G2Matrix1, model)
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
    ENN2 = totalEnergyScalar(G2Matrix2, model)
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
    println("Running pre-training Monte-Carlo simulation...\n")
    nsystems = length(systemParmsList)

    # Compute PMF for each system
    PMFs = []
    for systemId = 1:nsystems
        PMF = computePMF(refRDFs[systemId], systemParmsList[systemId], 1000)
        append!(PMFs, [PMF])
    end

    for step = 1:steps
        println("\nStep $(step)...\n")
        lossGradients = []
        for systemId = 1:nsystems
            # Read a frame from the trajectory
            traj = readXTC(systemParmsList[systemId])
            nframes = Int(size(traj)) - 1
            frameId = rand(rngXor, 1:nframes) # Don't take the first frame
            frame = read_step(traj, frameId)

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