using RandomNumbers

"""
function hist!(distanceMatrix, hist, systemParms)

Accumulates pair distances in a histogram
"""
function hist!(distanceMatrix, hist, systemParms)
    for i = 1:systemParms.N
        @fastmath for j = 1:i-1
            histIndex = floor(Int, 1 + distanceMatrix[i, j] / systemParms.binWidth)
            if histIndex <= systemParms.Nbins
                hist[histIndex] += 1
            end
        end
    end
    return (hist)
end

"""
function normalizehist!(hist, systemParms, box)

Normalizes distance histogram to RDF
"""
function normalizehist!(hist, systemParms, box)
    boxVolume = box[1] * box[2] * box[3]
    Npairs::Int = systemParms.N * (systemParms.N - 1) / 2
    bins = [bin * systemParms.binWidth for bin = 1:systemParms.Nbins]
    shellVolumes = [4 * π * systemParms.binWidth * bins[i]^2 for i in eachindex(bins)]
    rdfNorm = ones(Float64, systemParms.Nbins)
    for i in eachindex(rdfNorm)
        rdfNorm[i] = boxVolume / Npairs / shellVolumes[i]
    end
    hist .*= rdfNorm
    return (hist)
end

"""
function wrapFrame!(frame, box, pointIndex)

Returns all of the atoms into the original PBC image
"""
function wrapFrame!(frame, box, pointIndex)
    if (positions(frame)[1, pointIndex] < 0.0)
        positions(frame)[1, pointIndex] += box[1]
    end
    if (positions(frame)[1, pointIndex] > box[1])
        positions(frame)[1, pointIndex] -= box[1]
    end
    if (positions(frame)[2, pointIndex] < 0.0)
        positions(frame)[2, pointIndex] += box[2]
    end
    if (positions(frame)[2, pointIndex] > box[2])
        positions(frame)[2, pointIndex] -= box[2]
    end
    if (positions(frame)[3, pointIndex] < 0.0)
        positions(frame)[3, pointIndex] += box[3]
    end
    if (positions(frame)[3, pointIndex] > box[3])
        positions(frame)[3, pointIndex] -= box[3]
    end
end

"""
function atomicEnergy(inputlayer, model)

Computes the potential energy of one particle
from the input layer of the neural network
"""
function atomicEnergy(inputlayer, model)
    E::Float64 = model(inputlayer)[1]
    return (E)
end

"""
function totalEnergyScalar(symmFuncMatrix, model)

Computes the total potential energy of the system
"""
function totalEnergyScalar(symmFuncMatrix, model)
    N = size(symmFuncMatrix)[1]
    E = 0.0
    for i = 1:N
        E += atomicEnergy(symmFuncMatrix[i, :], model)
    end
    return (E)
end

"""
function getBoolMaskForUpdating(distanceVectorInput, NNParms)

Return boolean mask array of indexes for updating energies
True if the distance between moved particle and i-th atom is less than NNParms.maxDistanceCutoff
"""
function getBoolMaskForUpdating(distanceVectorInput, NNParms)
    N = size(distanceVectorInput)[1]
    indexes = Array{Bool}(undef, N)
    for i = 1:N
        indexes[i] = (distanceVectorInput[i] < NNParms.maxDistanceCutoff)
    end
    return (indexes)
end

"""
function totalEnergyVector(symmFuncMatrix, model, indexesForUpdate, previousE)

Computes vector of atomic energy contributions
"""
function totalEnergyVector(symmFuncMatrix, model, indexesForUpdate, previousE)
    N = size(symmFuncMatrix)[1]
    E = copy(previousE)
    for i = 1:N
        if indexesForUpdate[i]
            E[i] = atomicEnergy(symmFuncMatrix[i, :], model)
        end
    end
    return (E)
end

"""
function totalEnergyVectorInit(symmFuncMatrix, model)

Computes initial vector of energies for the system
"""
function totalEnergyVectorInit(symmFuncMatrix, model)
    N = size(symmFuncMatrix)[1]
    E = Array{Float64}(undef, N)
    for i = 1:N
        E[i] = atomicEnergy(symmFuncMatrix[i, :], model)
    end
    return (E)
end

"""
function mcmove!(mcarrays, E, model, NNParms, systemParms, rng)

Performs a Metropolis Monte Carlo
displacement move using a neural network
to predict energies from the symmetry function matrix
"""
function mcmove!(
    mcarrays,
    E,
    EpreviousVector,
    model,
    NNParms,
    systemParms,
    box,
    rng,
    mutatedStepAdjust,
)
    # Unpack mcarrays
    frame, distanceMatrix, G2Matrix1, G3Matrix1, G9Matrix1 = mcarrays
    # Optionally make a copy of the original coordinates
    if G3Matrix1 != [] || G9Matrix1 != []
        coordinates1 = copy(positions(frame))
    end

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(systemParms.N))

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Take a copy of the previous energy value
    E1 = copy(E)

    # Displace the particle
    dr = [
        mutatedStepAdjust * (rand(rng, Float64) - 0.5),
        mutatedStepAdjust * (rand(rng, Float64) - 0.5),
        mutatedStepAdjust * (rand(rng, Float64) - 0.5),
    ]

    positions(frame)[:, pointIndex] .+= dr

    # Check if all coordinates inside simulation box with PBC
    wrapFrame!(frame, box, pointIndex)

    # Compute the updated distance vector
    point = positions(frame)[:, pointIndex]
    distanceVector2 = computeDistanceVector(point, positions(frame), box)

    # Acceptance counter
    accepted = 0

    # Get indexes of atoms for energy contribution update
    indexesForUpdate = getBoolMaskForUpdating(distanceVector2, NNParms)

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

    # Make a copy of the original angular matrices and update them
    G3Matrix2 = copy(G3Matrix1)
    if G3Matrix1 != []
        updateG3Matrix!(
            G3Matrix2,
            coordinates1,
            positions(frame),
            box,
            distanceVector1,
            distanceVector2,
            systemParms,
            NNParms,
            pointIndex,
        )
    end

    G9Matrix2 = copy(G9Matrix1)
    if G9Matrix1 != []
        updateG9Matrix!(
            G9Matrix2,
            coordinates1,
            positions(frame),
            box,
            distanceVector1,
            distanceVector2,
            systemParms,
            NNParms,
            pointIndex,
        )
    end

    # Combine symmetry function matrices accumulators
    symmFuncMatrix2 = combineSymmFuncMatrices(G2Matrix2, G3Matrix2, G9Matrix2)

    # Compute the energy again
    # E2 = totalEnergyScalar(G2Matrix2, model) 
    newEnergyVector = totalEnergyVector(symmFuncMatrix2, model, indexesForUpdate, EpreviousVector)
    E2 = sum(newEnergyVector)

    # Get energy difference
    ΔE = E2 - E1

    # Accept or reject the move
    if rand(rng, Float64) < exp(-ΔE * systemParms.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector2
        distanceMatrix[:, pointIndex] = distanceVector2
        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix2, G3Matrix2, G9Matrix2)
        return (mcarrays, E, newEnergyVector, accepted)
    else
        positions(frame)[:, pointIndex] .-= dr

        # Check if all coordinates inside simulation box with PBC
        wrapFrame!(frame, box, pointIndex)

        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix1, G3Matrix1, G9Matrix1)
        return (mcarrays, E, EpreviousVector, accepted)
    end

end

"""
function mcsample!(input)

(input = model, globalParms, MCParms, NNParms, systemParms)
Runs the Monte Carlo simulation for a given
input configuration, set of parameters
and the neural network model
"""
function mcsample!(input)
    # Unpack the inputs
    model = input.model
    globalParms = input.globalParms
    MCParms = input.MCParms
    NNParms = input.NNParms
    systemParms = input.systemParms

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

    G3Matrix = []
    if length(NNParms.G3Functions) > 0
        G3Matrix = buildG3Matrix(distanceMatrix, positions(frame), box, NNParms)
    end

    G9Matrix = []
    if length(NNParms.G9Functions) > 0
        G9Matrix = buildG9Matrix(distanceMatrix, positions(frame), box, NNParms)
    end

    # Prepare a tuple of arrays that change duing the mcmove!
    mcarrays = (frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix)

    # Initialize the distance histogram accumulator
    histAccumulator = zeros(Float64, systemParms.Nbins)

    # Build the cross correlation arrays for training,
    # an additional distance histogram array
    # and the symmetry function matrix accumulator
    if globalParms.mode == "training"
        hist = zeros(Float64, systemParms.Nbins)
        G2MatrixAccumulator = zeros(size(G2Matrix))
        G3MatrixAccumulator = zeros(size(G3Matrix))
        G9MatrixAccumulator = zeros(size(G9Matrix))
        crossAccumulators = crossAccumulatorsInit(NNParms, systemParms)
    end

    # Combine symmetry function matrices
    symmFuncMatrix = combineSymmFuncMatrices(G2Matrix, G3Matrix, G9Matrix)

    # Initialize the starting energy and the energy array
    EpreviousVector = totalEnergyVectorInit(symmFuncMatrix, model)
    E = sum(EpreviousVector)
    energies = zeros(totalDataPoints + 1)
    energies[1] = E

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @fastmath for step = 1:MCParms.steps

        mcarrays, E, EpreviousVector, accepted = mcmove!(
            mcarrays,
            E,
            EpreviousVector,
            model,
            NNParms,
            systemParms,
            box,
            rngXor,
            mutatedStepAdjust,
        )
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        # Perform MC step adjustment during the equilibration
        if MCParms.stepAdjustFreq > 0 &&
           step % MCParms.stepAdjustFreq == 0 &&
           step < MCParms.Eqsteps
            mutatedStepAdjust = stepAdjustment!(
                mutatedStepAdjust,
                systemParms,
                box,
                MCParms,
                acceptedIntermediate,
            )
            acceptedIntermediate = 0
        end

        # Collect the output energies
        if step % MCParms.outfreq == 0
            energies[Int(step / MCParms.outfreq)+1] = E
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
    frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix = mcarrays # might remove this line
    if globalParms.mode == "training"
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
        G2MatrixAccumulator ./= prodDataPoints
        if G3Matrix != []
            G3MatrixAccumulator ./= prodDataPoints
        end
        if G9Matrix != []
            G9MatrixAccumulator ./= prodDataPoints
        end
        symmFuncMatrixAccumulator = combineSymmFuncMatrices(G2MatrixAccumulator, G3MatrixAccumulator, G9MatrixAccumulator)
    end

    # Normalize the sampled distance histogram
    histAccumulator ./= prodDataPoints
    normalizehist!(histAccumulator, systemParms, box)

    # Combine symmetry function matrices accumulators
    if globalParms.mode == "training"
        MCoutput = MCAverages(
            histAccumulator,
            energies,
            crossAccumulators,
            symmFuncMatrixAccumulator,
            acceptanceRatio,
            systemParms,
            mutatedStepAdjust,
        )
        return (MCoutput)
    else
        MCoutput = MCAverages(
            histAccumulator,
            energies,
            nothing,
            nothing,
            acceptanceRatio,
            systemParms,
            mutatedStepAdjust,
        )
        return (MCoutput)
    end
end

"""
function stepAdjustment!(systemParms, MCParms, acceptedIntermediate)

MC step length adjustment
"""
function stepAdjustment!(mutatedStepAdjust, systemParms, box, MCParms, acceptedIntermediate)
    acceptanceRatio = acceptedIntermediate / MCParms.stepAdjustFreq
    mutatedStepAdjust = acceptanceRatio * mutatedStepAdjust / systemParms.targetAR

    if mutatedStepAdjust > box[1]
        mutatedStepAdjust /= 2
    end

    return mutatedStepAdjust
end
