"""
function computePreTrainingLossGradients(energyPMF::Float64, symmFuncMatrix, model)

Computes loss gradients for one frame
"""
function computePreTrainingLossGradients(energyPMF::Float64, symmFuncMatrix, model, NNParms)
    N = size(symmFuncMatrix)[1]
    energyGradients = computeEnergyGradients(symmFuncMatrix, model)
    E = totalEnergyScalar(symmFuncMatrix, model)
    parameters = Flux.params(model)
    loss = (E - energyPMF)^2 / N
    regloss = sum(parameters[1].^2) * NNParms.REGP
    println("Energy loss (per atom): $(round(loss, digits=4))")
    println("PMF energy: $(round(energyPMF, digits=4))")
    println("NN energy: $(round(E, digits=4))")
    println("Regularization loss: $(round(regloss, digits=4))")
    gradientScaling::Float64 = 2 / N * (E - energyPMF)
    
    lossGradient = gradientScaling .* energyGradients
    regLossGradient = @. parameters * 2 * NNParms.REGP
    lossGradient += regLossGradient
    return (lossGradient)
end

"""
function computeMeanForcePotential(refRDF, systemParms)

Compute PMF energies for all the frames in a trajectory
"""
function computeMeanForcePotential(refRDF, systemParms, distanceScaling::Float64)
    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1
    energiesPMF = zeros(Float64, nframes)

    for frameId = 1:nframes
        frame = read_step(traj, frameId)
        distanceMatrix = buildDistanceMatrix(frame) 
        hist = zeros(Float64, systemParms.Nbins)
        hist = hist!(distanceMatrix .* distanceScaling, hist, systemParms)
        refRDF[refRDF .== 0] .= 1E-10
        PMF = -log.(refRDF) / systemParms.β
        E = sum(hist .* PMF)
        energiesPMF[frameId] = E
    end

    return (energiesPMF)
end

"""
function preTrain!(NNParms, systemParmsList, model, opt, refRDFs)

Runs pre-training using PMF as energy reference data.
    
The loss gradients are computed from one reference frame
and averaged over different systems.
"""
function preTrain!(NNParms, systemParmsList, model, opt, refRDFs)
    println("Running pre-training...\n")
    nsystems = length(systemParmsList)
    
    nframesMultiReference = []
    for systemId = 1:nsystems
        traj = readXTC(systemParmsList[systemId])
        nframes = Int(size(traj)) - 1
        append!(nframesMultiReference, nframes)
    end

    @assert length(unique(nframesMultiReference)) == 1 "Lengths of trajectories are different"
    nframes = nframesMultiReference[1]

    distanceScaling = LinRange(1.0, 0.1, 100)
    for scaling in distanceScaling
        energiesPMFMultiReference = []
        println("Scaling distances by $(round(scaling, digits=4))...\n")
        for systemId = 1:nsystems
            energiesPMF = computeMeanForcePotential(refRDFs[systemId], systemParmsList[systemId], scaling)
            append!(energiesPMFMultiReference, [energiesPMF])
        end

        for frameId = 1:nframes
            println("\nPre-training iteration $(frameId)...")
            lossGradients = []
            for systemId = 1:nsystems
                energyPMF = energiesPMFMultiReference[systemId][frameId]
                traj = readXTC(systemParmsList[systemId])
                frame = read_step(traj, frameId)
                distanceMatrix = buildDistanceMatrix(frame)
                G2Matrix = buildG2Matrix(distanceMatrix .* scaling, NNParms)
                lossGradient = computePreTrainingLossGradients(energyPMF, G2Matrix, model, NNParms)
                append!(lossGradients, [lossGradient])
            end
            meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
            updatemodel!(model, opt, meanLossGradients)
        end
    end
    @save "model-pre-trained.bson" model
    return (model, opt)
end
