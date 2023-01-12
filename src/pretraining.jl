"""
function computePMF(refRDF, systemParms)

Compute PMF for a given system (in kT units)
"""
function computePMF(refRDF, systemParms)
    PMF = Array{Float64}(undef, systemParms.Nbins)
    repulsionRegion = refRDF .== 0
    repulsionPoints = length(repulsionRegion[repulsionRegion.!=0])
    maxPMFIndex = repulsionPoints + 1
    maxPMF = -log(refRDF[maxPMFIndex]) / systemParms.β
    secondMaxPMF = -log(refRDF[maxPMFIndex+1]) / systemParms.β
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

function updatehist!(hist, distanceVector1, distanceVector2, systemParms)
    @fastmath for i = 1:systemParms.N
        # Remove histogram entries corresponding to distanceVector1 
        histIndex = floor(Int, 1 + distanceVector1[i] / systemParms.binWidth)
        if histIndex <= systemParms.Nbins
            hist[histIndex] -= 1
        end
        # Add histogram entries corresponding to distanceVector2
        histIndex = floor(Int, 1 + distanceVector2[i] / systemParms.binWidth)
        if histIndex <= systemParms.Nbins
            hist[histIndex] += 1
        end
    end
    return (hist)
end

"""
function computePMFEnergy(PMF, distanceMatrix, systemParms)

Compute a total energy of a configuration with PMF
"""
#function computePMFEnergy(PMF, distanceMatrix, systemParms)
#    hist = zeros(Float64, systemParms.Nbins)
#    hist = hist!(distanceMatrix, hist, systemParms)
#    E = sum(hist .* PMF)
#    return (E)
#end

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

    lossGradient = @. gradientScaling * (ENN2Gradients - ENN1Gradients)
    regLossGradient = @. parameters * 2 * NNParms.REGP
    lossGradient += regLossGradient
    return (lossGradient)
end

"""
function pretrainingMove!(frameInputArrays, PMF, model, NNParms, systemParms, rng)

Displaces one randomly selected particle,
computes energy differences using PMF and the neural network
"""
function pretrainingMove!(frameInputArrays, PMF, model, NNParms, systemParms, rng)
    # Unpack frame input arrays
    frame, distanceMatrix, hist, G2Matrix1 = frameInputArrays

    # Compute energy of the initial configuration
    ENN1Vector = totalEnergyVectorInit(G2Matrix1, model)
    ENN1 = sum(ENN1Vector)
    EPMF1 = sum(hist .* PMF)

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

    # Update the histogram
    hist = updatehist!(hist, distanceVector1, distanceVector2, systemParms)

    # Get indexes for updating ENN
    indexesForUpdate = getBoolMaskForUpdating(distanceVector2, systemParms, NNParms)

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

    # Compute the NN energy again
    ENN2Vector = totalEnergyVector(G2Matrix2, model, indexesForUpdate, ENN1Vector)
    ENN2 = sum(ENN2Vector)
    EPMF2 = sum(hist .* PMF)

    # Get the energy differences
    ΔENN = ENN2 - ENN1
    ΔEPMF = EPMF2 - EPMF1

    # Revert the changes in the frame arrays
    positions(frame)[:, pointIndex] .-= dr
    hist = updatehist!(hist, distanceVector2, distanceVector1, systemParms)

    # Pack frame output arrays
    frameOutputArrays = (G2Matrix1, G2Matrix2)

    return (frameOutputArrays, ΔENN, ΔEPMF)
end

"""
function preComputeFrameArrays(NNParms, systemParmsList, refRDFs)

Pre-compute PMFs, distance and G2 matrices as well as the reference trajectory frames
for later use in pre-training steps
"""
function preComputeFrameArrays(NNParms, systemParmsList, refRDFs)
    nsystems = length(systemParmsList)
    PMFs = [] # [nsystems]
    nframesPerSystem = [] # [nsystems]
    frames = [[] for i=1:nsystems] # [nsystems x nframes]
    distanceMatrices = [[] for i=1:nsystems] # [nsystems x nframes]
    histograms = [[] for i=1:nsystems] # [nsystems x nframes]
    G2Matrices = [[] for i=1:nsystems] # [nsystems x nframes]

    for systemId = 1:nsystems
        println("Pre-computing distances and G2 matrices for $(systemParmsList[systemId].systemName)...\n")
        PMF = computePMF(refRDFs[systemId], systemParmsList[systemId])
        append!(PMFs, [PMF])

        traj = readXTC(systemParmsList[systemId])
        nframes = Int(size(traj)) - 1 # Don't take the first frame
        append!(nframesPerSystem, nframes)
        if length(nframesPerSystem) > 1
            @assert nframesPerSystem[systemId] == nframesPerSystem[systemId-1]
        end

        for frameId = 1:nframes
            println("Frame $(frameId)...")
            frame = read_step(traj, frameId)
            append!(frames[systemId], [frame])

            distanceMatrix = buildDistanceMatrix(frame)
            append!(distanceMatrices[systemId], [distanceMatrix])

            hist = zeros(Float64, systemParmsList[systemId].Nbins)
            hist = hist!(distanceMatrix, hist, systemParmsList[systemId])
            append!(histograms[systemId], [hist])

            G2Matrix = buildG2Matrix(distanceMatrix, NNParms)
            append!(G2Matrices[systemId], [G2Matrix])
        end
    end
    frameArrays = (PMFs, frames, distanceMatrices, histograms, G2Matrices)
    return (frameArrays)
end

"""
function preTrain!(NNParms, systemParmsList, model, opt, frameArrays)

Run pre-training for a given number of steps
"""
function preTrain!(NNParms, systemParmsList, model, opt, frameArrays)
    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    println("Running $(NNParms.preTrainSteps) steps of pre-training Monte-Carlo...\n")
    nsystems = length(systemParmsList)

    # Unpack frameArrays from the preComputeFrameArrays function
    PMFs, frames, distanceMatrices, histograms, G2Matrices = frameArrays
    # Get the number of frames per system
    nframesPerSystem = Int(length(frames) / nsystems)

    for step = 1:NNParms.preTrainSteps
        println("\nStep $(step)...\n")
        lossGradients = []
        for systemId = 1:nsystems
            # ID of a frame within a system
            frameId = rand(rngXor, 1:nframesPerSystem)
            # Pack frame input arrays
            frameInputArrays = (
                frames[systemId][frameId],
                distanceMatrices[systemId][frameId],
                histograms[systemId][frameId],
                G2Matrices[systemId][frameId])

            # Run a pre-training step, compute energy differences with PMF and the neural network,
            # restore all input arguments to their original state
            frameOutputArrays, ΔENN, ΔEPMF = pretrainingMove!(
                frameInputArrays, PMFs[systemId], model, NNParms, systemParmsList[systemId], rngXor)
            G2Matrix1, G2Matrix2 = frameOutputArrays
            # Compute the loss gradient
            lossGradient = computePreTrainingLossGradients(
                ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, NNParms)
            append!(lossGradients, [lossGradient])
        end
        # Update the model
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        updatemodel!(model, opt, meanLossGradients)
    end
    @save "model-pre-trained.bson" model
    return (model)
end