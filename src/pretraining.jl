"""
struct referenceData

Contains pre-computed data for the reference trajectory frames

Fields:
    distanceMatrices: list of distance matrices of each frame
    histograms: list of distance histograms of each frame
    PMF: potential of mean force
    G2Matrices: list of G2 matrices of each frame
"""
struct referenceData
    distanceMatrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    PMF::Vector{Float64}
    G2Matrices::Vector{Matrix{Float64}}
end

"""
struct preComputeInput

Contains input data necessary for pre-computation
"""
struct preComputeInput
    NNParms::NNparameters
    systemParms::systemParameters
    refRDF::Vector{Float64}
end

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
function preComputeRefData(refDataInput)

Pre-compute PMFs, distance and G2 matrices as well as the reference trajectory frames
for later use in pre-training steps for a given system
"""
function preComputeRefData(refDataInput::preComputeInput)
    distanceMatrices = []
    histograms = []
    G2Matrices = []

    # Unpack the input struct
    NNParms = refDataInput.NNParms
    systemParms = refDataInput.systemParms
    refRDF = refDataInput.refRDF

    println("Pre-computing distances and G2 matrices for $(systemParms.systemName)...")
    PMF = computePMF(refRDF, systemParms)

    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame

    for frameId = 1:nframes
        #println("Frame $(frameId)...")
        frame = read_step(traj, frameId)

        distanceMatrix = buildDistanceMatrix(frame)
        append!(distanceMatrices, [distanceMatrix])

        hist = zeros(Float64, systemParms.Nbins)
        hist = hist!(distanceMatrix, hist, systemParms)
        append!(histograms, [hist])

        G2Matrix = buildG2Matrix(distanceMatrix, NNParms)
        append!(G2Matrices, [G2Matrix])
    end
    # Save the output in the referenceData struct
    refData = referenceData(distanceMatrices, histograms, PMF, G2Matrices)
    return (refData)
end

"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, preTrainParms, verbose=false)

Computes loss gradients for one frame
"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, G2Matrix1, G2Matrix2,
    model, preTrainParms::preTrainParameters, verbose=false)
    parameters = Flux.params(model)
    loss = (ΔENN - ΔEPMF)^2
    regloss = sum(parameters[1] .^ 2) * preTrainParms.PTREGP
    if verbose
        println("   Energy loss: $(round(loss, digits=4))")
        println("   PMF energy difference: $(round(ΔEPMF, digits=4))")
        println("   NN energy difference: $(round(ΔENN, digits=4))")
        println("   Regularization loss: $(round(regloss, digits=4))")
    end

    # Compute dL/dw
    ENN1Gradients = computeEnergyGradients(G2Matrix1, model)
    ENN2Gradients = computeEnergyGradients(G2Matrix2, model)
    gradientScaling = 2 * (ΔENN - ΔEPMF)

    lossGradient = @. gradientScaling * (ENN2Gradients - ENN1Gradients)
    regLossGradient = @. parameters * 2 * preTrainParms.PTREGP
    lossGradient += regLossGradient
    return (lossGradient)
end

"""
function pretrainingMove!(refData, model, NNParms, systemParms, rng)

Displaces one randomly selected particle,
computes energy differences using PMF and the neural network
"""
function pretrainingMove!(refData::referenceData, model, NNParms, systemParms, rng)
    # Pick a frame
    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame
    frameId = rand(rng, 1:nframes)
    frame = read_step(traj, frameId)
    # Pick a particle
    pointIndex = rand(rng, 1:systemParms.N)

    # Read reference data

    distanceMatrix = refData.distanceMatrices[frameId]
    hist = refData.histograms[frameId]
    G2Matrix1 = refData.G2Matrices[frameId]
    PMF = refData.PMF

    # Compute energy of the initial configuration
    ENN1Vector = totalEnergyVectorInit(G2Matrix1, model)
    ENN1 = sum(ENN1Vector)
    EPMF1 = sum(hist .* PMF)

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

    return (G2Matrix1, G2Matrix2, ΔENN, ΔEPMF)
end

"""
function preTrain!(preTrainParms, NNParms, systemParmsList, model, opt, refRDFs)

Run pre-training for a given number of steps
"""
function preTrain!(preTrainParms::preTrainParameters, NNParms, systemParmsList, model, opt, refRDFs)
    println("Running $(preTrainParms.PTsteps) steps of pre-training Monte-Carlo...\n")
    println("Neural network regularization parameter: $(preTrainParms.PTREGP)")
    println("Optimizer type: $(preTrainParms.PToptimizer)")
    println("Parameters of optimizer:")
    println("Learning rate: $(preTrainParms.PTrate)")

    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    nsystems = length(systemParmsList)

    # Pre-compute reference data in parallelrefRDFs
    refDataInputs = []
    for systemId = 1:nsystems
        refDataInput = preComputeInput(NNParms, systemParmsList[systemId], refRDFs[systemId])
        append!(refDataInputs, [refDataInput])
    end
    refDataList::Vector{referenceData} = pmap(preComputeRefData, refDataInputs)

    for step = 1:preTrainParms.PTsteps
        verbose::Bool = false
        if step % preTrainParms.PToutfreq == 0 || step == 1
            verbose = true
            println("\nStep $(step)...\n")
        end
        lossGradients = []
        for systemId = 1:nsystems
            # Pack frame input arrays   
            refData = refDataList[systemId]

            # Run a pre-training step, compute energy differences with PMF and the neural network,
            # restore all input arguments to their original state
            G2Matrix1, G2Matrix2, ΔENN, ΔEPMF = pretrainingMove!(
                refData, model, NNParms, systemParmsList[systemId], rngXor)

            # Compute the loss gradient
            lossGradient = computePreTrainingLossGradients(
                ΔENN, ΔEPMF, G2Matrix1, G2Matrix2, model, preTrainParms, verbose)
            append!(lossGradients, [lossGradient])
        end
        # Update the model
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        updatemodel!(model, opt, meanLossGradients)
    end
    @save "model-pre-trained.bson" model
    return (model)
end