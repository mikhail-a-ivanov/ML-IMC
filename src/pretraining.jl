"""
struct referenceData

Contains pre-computed data for the reference trajectory frames

Fields:
distanceMatrices: list of distance matrices of each frame
histograms: list of distance histograms of each frame
PMF: potential of mean force
G2Matrices: list of G2 matrices of each frame
G3Matrices: list of G3 matrices of each frame
G9Matrices: list of G9 matrices of each frame
"""
struct referenceData
    distanceMatrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    PMF::Vector{Float64}
    G2Matrices::Vector{Matrix{Float64}}
    G3Matrices::Vector{Matrix{Float64}}
    G9Matrices::Vector{Matrix{Float64}}
end

"""
struct preComputeInput

Contains input data necessary for pre-computation
"""
struct preComputeInput
    NNParms::NNparameters
    systemParms::SystemParameters
    refRDF::Vector{Float64}
end

"""
function computePMF(refRDF, systemParms)

Compute PMF for a given system (in kT units)
"""
function computePMF(refRDF, systemParms)
    PMF = Array{Float64}(undef, systemParms.Nbins)
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

function updatehist!(hist, distanceVector1, distanceVector2, systemParms)
    @fastmath for i in 1:(systemParms.N)
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

Pre-compute PMFs, distance and symmetry function matrices as well as the reference trajectory frames
for later use in pre-training steps for a given system
"""
function preComputeRefData(refDataInput::preComputeInput)::referenceData
    distanceMatrices = []
    histograms = []
    G2Matrices = []
    G3Matrices = []
    G9Matrices = []

    # Unpack the input struct
    NNParms = refDataInput.NNParms
    systemParms = refDataInput.systemParms
    refRDF = refDataInput.refRDF

    println("Pre-computing distances and symmetry function matrices for $(systemParms.systemName)...")
    PMF = computePMF(refRDF, systemParms)

    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame

    for frameId in 1:nframes
        #println("Frame $(frameId)...")
        frame = read_step(traj, frameId)
        box = lengths(UnitCell(frame))
        coordinates = positions(frame)

        distanceMatrix = buildDistanceMatrix(frame)
        append!(distanceMatrices, [distanceMatrix])

        hist = zeros(Float64, systemParms.Nbins)
        hist = hist!(distanceMatrix, hist, systemParms)
        append!(histograms, [hist])

        G2Matrix = buildG2Matrix(distanceMatrix, NNParms)
        append!(G2Matrices, [G2Matrix])

        if length(NNParms.G3Functions) > 0
            G3Matrix = buildG3Matrix(distanceMatrix, coordinates, box, NNParms)
            append!(G3Matrices, [G3Matrix])
        end

        if length(NNParms.G9Functions) > 0
            G9Matrix = buildG9Matrix(distanceMatrix, coordinates, box, NNParms)
            append!(G9Matrices, [G9Matrix])
        end
    end
    # Save the output in the referenceData struct
    refData = referenceData(distanceMatrices, histograms, PMF, G2Matrices, G3Matrices, G9Matrices)
    return (refData)
end

"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, symmFuncMatrix1, symmFuncMatrix2, model, preTrainParms, verbose=false)

Computes loss gradients for one frame
"""
function computePreTrainingLossGradients(ΔENN, ΔEPMF, symmFuncMatrix1, symmFuncMatrix2, model,
                                         preTrainParms::PreTrainParameters, NNParms::NNparameters, verbose=false)
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
    ENN1Gradients = computeEnergyGradients(symmFuncMatrix1, model, NNParms)
    ENN2Gradients = computeEnergyGradients(symmFuncMatrix2, model, NNParms)
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
function pretrainingMoveSym!(refData::referenceData, model, NNParms, systemParms, rng)
    # Pick a frame
    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame
    frameId = rand(rng, 1:nframes)
    frame = read_step(traj, frameId)
    box = lengths(UnitCell(frame))
    # Pick a particle
    pointIndex = rand(rng, 1:(systemParms.N))

    # Read reference data
    distanceMatrix = refData.distanceMatrices[frameId]
    hist = refData.histograms[frameId]
    PMF = refData.PMF

    # If no angular symmetry functions are provided, use G2 only
    if refData.G3Matrices == [] && refData.G9Matrices == []
        symmFuncMatrix1 = refData.G2Matrices[frameId]
    else
        # Make a copy of the original coordinates
        coordinates1 = copy(positions(frame))
        # Combine symmetry function matrices
        if refData.G3Matrices == []
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G9Matrices[frameId]]
        elseif refData.G9Matrices == []
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G3Matrices[frameId]]
        else
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G3Matrices[frameId], refData.G9Matrices[frameId]]
        end
        # Unpack symmetry functions and concatenate horizontally into a single matrix
        symmFuncMatrix1 = hcat(symmFuncMatrices...)
    end

    # Compute energy of the initial configuration
    ENN1Vector = totalEnergyVectorInit(symmFuncMatrix1, model)
    ENN1 = sum(ENN1Vector)
    EPMF1 = sum(hist .* PMF)

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Displace the particle
    dr = [systemParms.Δ * (rand(rng, Float64) - 0.5), systemParms.Δ * (rand(rng, Float64) - 0.5),
          systemParms.Δ * (rand(rng, Float64) - 0.5)]

    positions(frame)[:, pointIndex] .+= dr

    # Compute the updated distance vector
    point = positions(frame)[:, pointIndex]
    distanceVector2 = computeDistanceVector(point, positions(frame), box)

    # Update the histogram
    hist = updatehist!(hist, distanceVector1, distanceVector2, systemParms)

    # Get indexes for updating ENN
    indexesForUpdate = getBoolMaskForUpdating(distanceVector2, NNParms)

    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(refData.G2Matrices[frameId])
    updateG2Matrix!(G2Matrix2, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

    # Make a copy of the original angular matrices and update them
    G3Matrix2 = []
    if refData.G3Matrices != []
        G3Matrix2 = copy(refData.G3Matrices[frameId])
        updateG3Matrix!(G3Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Make a copy of the original angular matrices and update them
    G9Matrix2 = []
    if refData.G9Matrices != []
        G9Matrix2 = copy(refData.G9Matrices[frameId])
        updateG9Matrix!(G9Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Combine symmetry function matrices accumulators
    symmFuncMatrix2 = combineSymmFuncMatrices(G2Matrix2, G3Matrix2, G9Matrix2)

    # Compute the NN energy again
    ENN2Vector = totalEnergyVector(symmFuncMatrix2, model, indexesForUpdate, ENN1Vector)
    ENN2 = sum(ENN2Vector)
    EPMF2 = sum(hist .* PMF)

    # Get the energy differences
    ΔENN = ENN2 - ENN1
    ΔEPMF = EPMF2 - EPMF1

    # Revert the changes in the frame arrays
    positions(frame)[:, pointIndex] .-= dr
    hist = updatehist!(hist, distanceVector2, distanceVector1, systemParms)

    return (symmFuncMatrix1, symmFuncMatrix2, ΔENN, ΔEPMF)
end

"""
function preTrain!(preTrainParms, NNParms, systemParmsList, model, opt, refRDFs)

Run pre-training for a given number of steps
"""
function preTrainSymFun!(preTrainParms::PreTrainParameters, NNParms, systemParmsList, model, opt, refRDFs)
    println("\nRunning $(preTrainParms.PTsteps) steps of pre-training Monte-Carlo...\n")
    println("Neural network regularization parameter: $(preTrainParms.PTREGP)")
    reportOpt(opt)

    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    nsystems = length(systemParmsList)

    # Pre-compute reference data in parallelrefRDFs
    refDataInputs = []
    for systemId in 1:nsystems
        refDataInput = preComputeInput(NNParms, systemParmsList[systemId], refRDFs[systemId])
        append!(refDataInputs, [refDataInput])
    end
    refDataList::Vector{referenceData} = pmap(preComputeRefData, refDataInputs)

    for step in 1:(preTrainParms.PTsteps)
        verbose::Bool = false
        if step % preTrainParms.PToutfreq == 0 || step == 1
            verbose = true
            println("\nStep $(step)...\n")
        end
        lossGradients = []
        for systemId in 1:nsystems
            # Pack frame input arrays   
            refData = refDataList[systemId]

            # Run a pre-training step, compute energy differences with PMF and the neural network,
            # restore all input arguments to their original state
            symmFuncMatrix1, symmFuncMatrix2, ΔENN, ΔEPMF = pretrainingMoveSym!(refData, model, NNParms,
                                                                                systemParmsList[systemId], rngXor)

            # Compute the loss gradient
            lossGradient = computePreTrainingLossGradients(ΔENN, ΔEPMF, symmFuncMatrix1, symmFuncMatrix2, model,
                                                           preTrainParms, NNParms, verbose)
            append!(lossGradients, [lossGradient])
        end
        # Update the model
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        updatemodel!(model, opt, meanLossGradients)
    end
    @save "model-pre-trained.bson" model
    checkfile("model-pre-trained.bson")
    return (model)
end
