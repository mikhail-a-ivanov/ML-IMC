mutable struct InputPreTrainData
    PMF::Vector{Float64}
    frame::Frame
    dist_mat::Matrix{Float64}
    histogram::Vector{Float64}
    g2_mat::Matrix{Float64}
    energy_vec::Vector{Float64}
end

struct MCSampleInputOther
    globalParms::GlobalParameters
    MCParms::MCparameters
    NNParms::NNparameters
    systemParms::SystemParameters
    model::Chain
    PMF::Vector{Float64}
end

function initHist(distanceMatrix, systemParms)
    hist = zeros(Float64, systemParms.Nbins)
    for i in 1:(systemParms.N)
        @fastmath for j in 1:(i - 1)
            histIndex = floor(Int, 1 + distanceMatrix[i, j] / systemParms.binWidth)
            if histIndex <= systemParms.Nbins
                hist[histIndex] += 1
            end
        end
    end
    return hist
end

function initHistForVec(dist_vec, systemParms)
    hist = zeros(Float64, systemParms.Nbins)
    for i in 1:(systemParms.N)
        histIndex = floor(Int, 1 + dist_vec[i] / systemParms.binWidth)
        if histIndex <= systemParms.Nbins
            hist[histIndex] += 1
        end
    end
    return hist
end

function computePMFenergyVec(dist_mat, pmf_system, systemParms)
    energy_vec = []
    for i in 1:(systemParms.N)
        dist_vec_i = dist_mat[i, :]
        hist_i = initHistForVec(dist_vec_i, systemParms)
        E_pmf_i = sum(hist_i .* pmf_system)
        append!(energy_vec, E_pmf_i)
    end
    return energy_vec
end

function atomicEnergyNNOther(input_vec, model)
    E::Float64 = model(input_vec[1])[1]
    return (E)
end

function totalEnergyVectorInitOther(input_mat, model)
    N = size(input_mat)[1]
    E = Array{Float64}(undef, N)
    for i in 1:N
        E[i] = atomicEnergyNNOther(input_mat[i, :], model)
    end
    return (E)
end

function totalEnergyVectorUpdateOther(input_mat, model, indexesForUpdate, nn_evergy_prev_vec)
    N = size(input_mat)[1]
    E = copy(nn_evergy_prev_vec)
    for i in 1:N
        if indexesForUpdate[i]
            E[i] = atomicEnergyNNOther(input_mat[i, :], model)
        end
    end
    return (E)
end

function totalEnergyScalarOther(input_mat, model)
    N = size(input_mat)[1]
    E = 0.0
    for i in 1:N
        E += atomicEnergyNNOther(input_mat[i, :], model)
    end
    return (E)
end

function createInputMatOther(energy_vec, g2_mat, sysParams::SystemParameters)
    N = sysParams.N
    input_mat = Array{Vector{Float64}}(undef, N)
    for i in 1:N
        # PROBABLY slow operation!!!!!!!
        input_mat[i] = pushfirst!(g2_mat[i, :], sysParams.concentration, energy_vec[i])
    end
    return input_mat
end

function computeEnergyGradientsOther(input_mat, model, NNParms)
    energyGradients = []
    # Compute energy gradients
    gs = gradient(totalEnergyScalarOther, input_mat, model)
    # Loop over the gradients and collect them in the array
    # Structure: gs[2][1][layerId][1 - weigths; 2 - biases]
    for (layerId, layerGradients) in enumerate(gs[2][1])
        if NNParms.bias
            weightGradients = layerGradients[1]
            append!(energyGradients, [weightGradients])
            biasGradients = layerGradients[2]
            append!(energyGradients, [biasGradients])
        else
            weightGradients = layerGradients[1]
            append!(energyGradients, [weightGradients])
        end
    end
    return (energyGradients)
end

function computePreTrainingLossGradientsOther!(ΔENN, ΔEPMF, input_1, input_2, model, NNParms::NNparameters,
                                               verbose=false)
    # parameters = Flux.params(model)
    loss = (ΔENN - ΔEPMF)^2

    if verbose
        println("   Energy loss: $(round(loss, digits=8))")
        println("   PMF energy difference: $(round(ΔEPMF, digits=6))")
        println("   NN energy difference:  $(round(ΔENN, digits=6))")
    end

    # Compute dL/dw
    ENN1Gradients = computeEnergyGradientsOther(input_1, model, NNParms)
    ENN2Gradients = computeEnergyGradientsOther(input_2, model, NNParms)
    gradientScaling = 2 * (ΔENN - ΔEPMF)

    lossGradient = @. gradientScaling * (ENN2Gradients - ENN1Gradients)
    return (lossGradient)
end

function write_loss_files(losses_array, nsystems)
    for sysID in 1:nsystems
        # Open the file for writing
        filename = "pre_training_losses_$(sysID).txt"
        io = open(filename, "w")

        # Write column headers
        println(io, "# epoch     value")

        # Write data to the file
        for (epoch, loss_value) in enumerate(losses_array[sysID])
            println(io, "$epoch    $loss_value")
        end

        # Close the file
        close(io)
    end
end

# function pretrainingMoveOther!(refData::referenceData, model, NNParms, systemParms, rng)
function pretrainingMoveOther!(model, input_struct::InputPreTrainData, NNParms, systemParms, rng)
    # Unpacking of struct
    PMF = input_struct.PMF
    frame = deepcopy(input_struct.frame)
    dist_mat = copy(input_struct.dist_mat)
    hist = copy(input_struct.histogram)
    g2_mat = copy(input_struct.g2_mat)
    energy_vec = copy(input_struct.energy_vec)

    box = lengths(UnitCell(frame))

    # Pick a RaNDom Particle
    rndp = rand(rng, 1:(systemParms.N))

    # Displace the particle
    dr = systemParms.Δ * [(rand(rng, Float64) - 0.5), (rand(rng, Float64) - 0.5), (rand(rng, Float64) - 0.5)]

    # Make random shift
    positions(frame)[:, rndp] += dr

    # Update all changed things after step
    dist_mat[:, rndp] = updateDistance!(frame, dist_mat[:, rndp], rndp)
    hist = updatehist!(hist, input_struct.dist_mat[:, rndp], dist_mat[:, rndp], systemParms)
    g2_mat = updateG2Matrix!(g2_mat, input_struct.dist_mat[:, rndp], dist_mat[:, rndp], systemParms, NNParms, rndp)
    energy_vec_after = computePMFenergyVec(dist_mat, PMF, systemParms)

    # Compute all energies
    input_mat_to_nn = createInputMatOther(energy_vec, input_struct.g2_mat, systemParms)
    E_NN_before = totalEnergyVectorInitOther(input_mat_to_nn, model)

    input_mat_to_nn_after = createInputMatOther(energy_vec_after, g2_mat, systemParms)
    E_NN_after = totalEnergyVectorInitOther(input_mat_to_nn_after, model)

    ENN1 = sum(E_NN_before)
    ENN2 = sum(E_NN_after)
    ΔENN = ENN2 - ENN1

    EPMF1 = sum(input_struct.histogram .* PMF)
    EPMF2 = sum(hist .* PMF)
    ΔEPMF = EPMF2 - EPMF1

    # Accept or revert step
    if (ΔEPMF < 0.0 || rand(rng, Float64) < exp(-ΔEPMF * systemParms.β))
        input_struct.frame = frame
        input_struct.dist_mat = dist_mat
        input_struct.histogram = hist
        input_struct.g2_mat = g2_mat
        input_struct.energy_vec = energy_vec_after
    end

    # Return input for NN and delta Energies for Loss
    return (input_struct, input_mat_to_nn, input_mat_to_nn_after, ΔENN, ΔEPMF)
end

function preTrainOther!(preTrainParms::PreTrainParameters, NNParms, systemParmsList, model, opt, refRDFs)
    rng = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    nsystems = length(systemParmsList)

    # Init PMF functions CONST
    systems_pmf_list = []
    for systemID in 1:nsystems
        pmf = computePMF(refRDFs[systemID], systemParmsList[systemID])
        append!(systems_pmf_list, [pmf])
    end

    # Init start frames (positions) for all systems
    systems_frame_list = [] # start positions
    for systemID in 1:nsystems
        traj = readXTC(systemParmsList[systemID])
        frame = read(traj)
        append!(systems_frame_list, [frame])
    end

    # Compute start distance matrices for all systems
    systems_dist_mats = []
    for systemID in 1:nsystems
        dist_mat = buildDistanceMatrixChemfiles(systems_frame_list[systemID])
        append!(systems_dist_mats, [dist_mat])
    end

    # Init Histograms for start positions
    systems_hist_list = []
    for systemID in 1:nsystems
        hist = initHist(systems_dist_mats[systemID], systemParmsList[systemID])
        append!(systems_hist_list, [hist])
    end

    # Init vector of G2 matrices for systems
    systems_g2_mats = []
    for systemID in 1:nsystems
        g2_mat = buildG2Matrix(systems_dist_mats[systemID], NNParms)
        append!(systems_g2_mats, [g2_mat])
    end

    # Compute start energy vector for each particle and each system
    systems_energy_vec = []
    for systemID in 1:nsystems
        energy_vec = computePMFenergyVec(systems_dist_mats[systemID], systems_pmf_list[systemID],
                                         systemParmsList[systemID])
        append!(systems_energy_vec, [energy_vec])
    end

    # Creating structs 
    systems_input_struct_list = InputPreTrainData[] # Special constructor for list of structs
    for sysID in 1:nsystems
        input_struct = InputPreTrainData(systems_pmf_list[sysID], systems_frame_list[sysID], systems_dist_mats[sysID],
                                         systems_hist_list[sysID], systems_g2_mats[sysID], systems_energy_vec[sysID])
        push!(systems_input_struct_list, input_struct)
    end

    println("\nRunning $(preTrainParms.PTsteps) steps of pre-training Monte-Carlo...\n")
    reportOpt(opt)

    losses_array = Vector{Vector{Float64}}(undef, nsystems)
    for sysID in 1:nsystems
        losses_array[sysID] = [0.0]  # Initialize each element as an array with one element
    end

    # Main Loop with Pre-Training Iterations
    for epoch in 1:(preTrainParms.PTsteps)
        verbose::Bool = false
        if epoch % preTrainParms.PToutfreq == 0 || epoch == 1
            verbose = true
            println("\nEpoch $(epoch)...\n")
        end

        lossGradients = []
        for sysID in 1:nsystems
            systems_input_struct_list[sysID], input_mat_1, input_mat_2, ΔENN, ΔEPMF = pretrainingMoveOther!(model,
                                                                                                            systems_input_struct_list[sysID],
                                                                                                            NNParms,
                                                                                                            systemParmsList[sysID],
                                                                                                            rng)

            append!(losses_array[sysID], (ΔENN - ΔEPMF)^2)
            # Compute the loss gradient
            lossGradient = computePreTrainingLossGradientsOther!(ΔENN, ΔEPMF, input_mat_1, input_mat_2, model, NNParms,
                                                                 verbose)
            append!(lossGradients, [lossGradient])
        end
        # Update the model
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        updatemodel!(model, opt, meanLossGradients)
    end

    write_loss_files(losses_array, nsystems)
    @save "model-pre-trained.bson" model
    checkfile("model-pre-trained.bson")
    return (model)
end

function mcmoveOther!(mcarrays, E, nn_energy_vec_prev, model, NNParams, sysParams, box, rng, mut_step_adjst, PMF)
    # Unpack mcarrays
    frame, dist_mat, g2_mat_1 = mcarrays
    # Take a copy of the previous energy value
    E1 = copy(E)    # NN

    # Pick a particle
    rndp = rand(rng, Int32(1):Int32(sysParams.N))
    # Allocate the distance vector and Check if all coordinates inside simulation box with PBC
    dist_vec_1 = dist_mat[:, rndp]

    # Displace the particle
    dr = mut_step_adjst * [(rand(rng, Float64) - 0.5), (rand(rng, Float64) - 0.5), (rand(rng, Float64) - 0.5)]
    positions(frame)[:, rndp] .+= dr
    wrapFrame!(frame, box, rndp)

    # Compute the updated distance vector
    point = positions(frame)[:, rndp]
    dist_vec_2 = computeDistanceVector(point, positions(frame), box)
    # Get indexes of atoms for energy contribution update
    indexesForUpdate = getBoolMaskForUpdating(dist_vec_2, NNParams)
    # Make a copy of the original G2 matrix and update it
    g2_mat_2 = copy(g2_mat_1)
    g2_mat_2 = updateG2Matrix!(g2_mat_2, dist_vec_1, dist_vec_2, sysParams, NNParams, rndp)

    dist_mat[:, rndp] = dist_vec_2
    pmf_energy_vec_current = computePMFenergyVec(dist_mat, PMF, sysParams)
    # Combine symmetry function matrices accumulators
    input_mat_2 = createInputMatOther(pmf_energy_vec_current, g2_mat_2, sysParams)

    # Compute the energy again
    # E2 = totalEnergyScalar(G2Matrix2, model) 
    nn_evergy_vec_cur = totalEnergyVectorUpdateOther(input_mat_2, model, indexesForUpdate, nn_energy_vec_prev)
    E2 = sum(nn_evergy_vec_cur)

    # Get energy difference
    ΔE = E2 - E1    # NN 

    accepted = 0    # Acceptance counter
    # Accept or reject the move
    if rand(rng, Float64) < exp(-ΔE * sysParams.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        """
        Ужасно важное место! Надо проверить, чтобы в претрейнинге было так же!!!!
        """
        dist_mat[rndp, :] = dist_vec_2
        dist_mat[:, rndp] = dist_vec_2
        # Pack mcarrays
        mcarrays = (frame, dist_mat, g2_mat_2)
        return (mcarrays, E, nn_evergy_vec_cur, accepted)
    else
        positions(frame)[:, rndp] .-= dr

        # Check if all coordinates inside simulation box with PBC
        wrapFrame!(frame, box, rndp)

        # Pack mcarrays
        mcarrays = (frame, dist_mat, g2_mat_1)
        return (mcarrays, E, nn_energy_vec_prev, accepted)
    end
end

function prepMCInputsOther(globalParms, MCParms, NNParms, systemParmsList, model, refRDFs)
    nsystems = length(systemParmsList)
    multiReferenceInput = []
    for systemId in 1:nsystems
        pmf = computePMF(refRDFs[systemId], systemParmsList[systemId])
        input = MCSampleInputOther(globalParms, MCParms, NNParms, systemParmsList[systemId], model, pmf)
        append!(multiReferenceInput, [input])
    end
    nsets = Int(nworkers() / nsystems)
    inputs = []
    for setId in 1:nsets
        append!(inputs, multiReferenceInput)
    end
    return (inputs)
end

function mcsampleOther!(input::MCSampleInputOther)
    # Unpack the inputs
    model = input.model
    globalParms = input.globalParms
    MCParms = input.MCParms
    NNParms = input.NNParms
    systemParms = input.systemParms
    PMF = input.PMF

    mutatedStepAdjust = copy(systemParms.Δ)

    # Get the worker id and the output filenames
    if nprocs() == 1
        id = myid()
    else
        id = myid() - 1
    end
    idString = lpad(id, 3, '0')

    trajFile = "mctraj-p$(idString).xtc"
    pdbFile = "confin-p$(idString).pdb"

    # Initialize RNG
    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    if globalParms.mode == "training"
        # Take a random frame from the equilibrated trajectory
        traj = readXTC(systemParms)
        nframes = Int(size(traj)) - 1
        frameId = rand(rngXor, 1:nframes) # Don't take the first frame
        frame = deepcopy(read_step(traj, frameId))
    else
        # Read PDB data from the system.in file
        pdb = readPDB(systemParms)
        frame = deepcopy(read_step(pdb, 0))
    end

    # Get current box vectors
    box = lengths(UnitCell(frame))

    # Start writing MC trajectory
    if globalParms.outputMode == "verbose"
        writeTraj(positions(frame), box, systemParms, trajFile, 'w')
        writeTraj(positions(frame), box, systemParms, pdbFile, 'w')
    end

    # Get the number of data points
    totalDataPoints = Int(MCParms.steps / MCParms.outfreq)
    prodDataPoints = Int((MCParms.steps - MCParms.Eqsteps) / MCParms.outfreq)

    # Build the distance matrix
    distanceMatrix = buildDistanceMatrix(frame)

    # Build the symmetry function matrices
    G2Matrix = buildG2Matrix(distanceMatrix, NNParms)

    # Prepare a tuple of arrays that change duing the mcmove!
    mcarrays = (frame, distanceMatrix, G2Matrix)

    # Initialize the distance histogram accumulator
    histAccumulator = zeros(Float64, systemParms.Nbins)

    # Build the cross correlation arrays for training,
    # an additional distance histogram array
    # and the symmetry function matrix accumulator
    if globalParms.mode == "training"
        hist = zeros(Float64, systemParms.Nbins)
        G2MatrixAccumulator = zeros(size(G2Matrix))
        crossAccumulators = crossAccumulatorsInit(NNParms, systemParms)
    end

    # Combine symmetry function matrices
    # symmFuncMatrix = combineSymmFuncMatrices(G2Matrix, G3Matrix, G9Matrix)

    # Initialize the starting energy and the energy array
    pmf_energy_vec = computePMFenergyVec(distanceMatrix, PMF, systemParms)
    input_nn_mat = createInputMatOther(pmf_energy_vec, G2Matrix, systemParms)
    nn_energy_prev_vec = totalEnergyVectorInitOther(input_nn_mat, model)
    E = sum(nn_energy_prev_vec)
    energies = zeros(totalDataPoints + 1)
    energies[1] = E

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @fastmath for step in 1:(MCParms.steps)
        mcarrays, E, nn_energy_prev_vec, accepted = mcmoveOther!(mcarrays, E, nn_energy_prev_vec, model, NNParms,
                                                                 systemParms, box, rngXor, mutatedStepAdjust, PMF)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        # Perform MC step adjustment during the equilibration
        if MCParms.stepAdjustFreq > 0 && step % MCParms.stepAdjustFreq == 0 && step < MCParms.Eqsteps
            mutatedStepAdjust = stepAdjustment!(mutatedStepAdjust, systemParms, box, MCParms, acceptedIntermediate)
            acceptedIntermediate = 0
        end

        # Collect the output energies
        if step % MCParms.outfreq == 0
            energies[Int(step / MCParms.outfreq) + 1] = E
        end

        # MC trajectory output
        if globalParms.outputMode == "verbose"
            if step % MCParms.trajout == 0
                writeTraj(positions(mcarrays[1]), box, systemParms, trajFile, 'a')
            end
        end

        # Accumulate the distance histogram
        if step % MCParms.outfreq == 0 && step > MCParms.Eqsteps
            frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix = mcarrays
            # Update the cross correlation array during the training
            if globalParms.mode == "training"
                hist = hist!(distanceMatrix, hist, systemParms)
                histAccumulator .+= hist
                G2MatrixAccumulator .+= G2Matrix
                if G3Matrix != []
                    G3MatrixAccumulator .+= G3Matrix
                end
                if G9Matrix != []
                    G9MatrixAccumulator .+= G9Matrix
                end
                # Normalize the histogram to RDF
                normalizehist!(hist, systemParms, box)

                # Combine symmetry function matrices
                symmFuncMatrix = combineSymmFuncMatrices(G2Matrix, G3Matrix, G9Matrix)

                updateCrossAccumulators!(crossAccumulators, symmFuncMatrix, hist, model, NNParms)
                # Nullify the hist array for the next training iteration
                hist = zeros(Float64, systemParms.Nbins)
            else
                histAccumulator = hist!(distanceMatrix, histAccumulator, systemParms)
            end
        end
    end
    # Compute and report the final acceptance ratio
    acceptanceRatio = acceptedTotal / MCParms.steps

    # Unpack mcarrays and optionally normalize cross and G2Matrix accumulators
    frame, distanceMatrix, G2Matrix = mcarrays # might remove this line
    if globalParms.mode == "training"
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
        G2MatrixAccumulator ./= prodDataPoints

        symmFuncMatrixAccumulator = combineSymmFuncMatrices(G2MatrixAccumulator, G3MatrixAccumulator,
                                                            G9MatrixAccumulator)
    end

    # Normalize the sampled distance histogram
    histAccumulator ./= prodDataPoints
    normalizehist!(histAccumulator, systemParms, box)

    # Combine symmetry function matrices accumulators
    if globalParms.mode == "training"
        MCoutput = MCAverages(histAccumulator, energies, crossAccumulators, symmFuncMatrixAccumulator, acceptanceRatio,
                              systemParms, mutatedStepAdjust)
        return (MCoutput)
    else
        MCoutput = MCAverages(histAccumulator, energies, nothing, nothing, acceptanceRatio, systemParms,
                              mutatedStepAdjust)
        return (MCoutput)
    end
end

function trainOther!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    # Run training iterations
    iteration = 1

    while iteration <= NNParms.iters
        iterString = lpad(iteration, 2, '0')
        println("\nIteration $(iteration)...")

        # Prepare multi-reference inputs
        inputs = prepMCInputsOther(globalParms, MCParms, NNParms, systemParmsList, model, refRDFs)

        # Run the simulation in parallel
        outputs = pmap(mcsampleOther!, inputs)

        # Collect averages corresponding to each reference system
        systemOutputs, systemLosses = collectSystemAverages(outputs, refRDFs, systemParmsList, globalParms, NNParms,
                                                            model)

        # Compute loss and the gradients
        lossGradients = []
        for (systemId, systemOutput) in enumerate(systemOutputs)
            systemParms = systemParmsList[systemId]
            lossGradient = computeLossGradients(systemOutput.crossAccumulators, systemOutput.symmFuncMatrixAccumulator,
                                                systemOutput.descriptor, refRDFs[systemId], model, systemParms, NNParms)
            append!(lossGradients, [lossGradient])
            # Write descriptors and energies
            name = systemParms.systemName
            writeRDF("RDFNN-$(name)-iter-$(iterString).dat", systemOutput.descriptor, systemParms)
            # writeEnergies("energies-$(name)-iter-$(iterString).dat", systemOutput.energies, MCParms, systemParms, 1)
        end
        # Average the gradients
        if globalParms.adaptiveScaling
            gradientCoeffs = adaptiveGradientCoeffs(systemLosses)
            println("\nGradient scaling: \n")
            for (gradientCoeff, systemParms) in zip(gradientCoeffs, systemParmsList)
                println("   System $(systemParms.systemName): $(round(gradientCoeff, digits=8))")
            end

            meanLossGradients = sum(lossGradients .* gradientCoeffs)
        else
            meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        end

        # Write the model and opt (before training!) and the gradients
        @save "model-iter-$(iterString).bson" model
        checkfile("model-iter-$(iterString).bson")

        # @save "opt-iter-$(iterString).bson" opt
        # checkfile("opt-iter-$(iterString).bson")

        # @save "gradients-iter-$(iterString).bson" meanLossGradients
        # checkfile("gradients-iter-$(iterString).bson")

        # Update the model
        updatemodel!(model, opt, meanLossGradients)

        # Move on to the next iteration
        iteration += 1
    end
    println("The training is finished!")
    return
end
