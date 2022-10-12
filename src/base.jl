using Printf
using RandomNumbers
using Statistics
using StaticArrays
using LinearAlgebra
using Chemfiles
using Flux

include("distances.jl")
include("network.jl")
include("io.jl")

"""
function G2(R, Rc, Rs, η)

Computes a single exponent
of the G2 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function G2(R, Rc, Rs, η)
    return(exp(-η*(R - Rs)^2) * distanceCutoff(R, Rc))
end

"""
function G2total(distances, Rc, Rs, sigma)

Computes the total G2 symmetry function (J. Chem. Phys. 134, 074106 (2011))
"""
function G2total(distances, Rc, Rs, sigma)
    sum = 0
    η = 1/(2*sigma*sigma)
    @fastmath @inbounds @simd for R in distances
        sum += G2(R, Rc, Rs, η)
    end
    return(sum)
end

"""
function buildG2Matrix(distanceMatrix, NNParms)

Builds a matrix of G2 values with varying Rs and η parameters for each atom in the configuration
"""
function buildG2Matrix(distanceMatrix, NNParms)
    N = size(distanceMatrix)[1]
    npoints = NNParms.neurons[1]
    Rss = LinRange(NNParms.minR, NNParms.maxR, npoints)
    G2Matrix = zeros(Float64, N, npoints)
    for i in 1:N
        distanceVector = distanceMatrix[i, :]
        for j in 1:npoints
            G2Matrix[i, j] = G2total(distanceVector, NNParms.maxR, Rss[j], NNParms.sigma)
        end
    end
    return(G2Matrix)
end

"""
function updateG2Matrix!(G2Matrix, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

Updates the G2 matrix with the displacement of a single atom
"""
function updateG2Matrix!(G2Matrix, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)
    npoints = NNParms.neurons[1]
    Rss = LinRange(NNParms.minR, NNParms.maxR, npoints)
    for i in 1:systemParms.N
        # Rebuild the whole G2 matrix column for the displaced particle
        if i == pointIndex
            for j in 1:npoints
                G2Matrix[pointIndex, j] = G2total(distanceVector2, NNParms.maxR, Rss[j], NNParms.sigma)
            end
        # Compute the change in G2 caused by the displacement of an atom
        else
            if distanceVector2[i] < NNParms.maxR
                for j in 1:npoints
                    G2_1 = G2(distanceVector1[i], NNParms.maxR, Rss[j], NNParms.sigma)
                    G2_2 = G2(distanceVector2[i], NNParms.maxR, Rss[j], NNParms.sigma)
                    ΔG2 = G2_2 - G2_1
                    G2Matrix[i, j] += ΔG2
                end
            end
        end
    end
    return(G2Matrix)
end

"""
function hist!(distanceMatrix, hist, systemParms)

Accumulates pair distances in a histogram
"""
function hist!(distanceMatrix, hist, systemParms)
    @inbounds for i in 1:systemParms.N
        @inbounds @fastmath for j in 1:i-1
            histIndex = floor(Int, 0.5 + distanceMatrix[i,j]/systemParms.binWidth)
            if histIndex <= systemParms.Nbins
                hist[histIndex] += 1
            end
        end
    end
    return(hist)
end

"""
function normalizehist!(hist, systemParms)

Normalizes distance histogram to RDF
"""
function normalizehist!(hist, systemParms)
    Npairs::Int = systemParms.N*(systemParms.N-1)/2
    bins = [bin*systemParms.binWidth for bin in 1:systemParms.Nbins]
    shellVolumes = [4*π*systemParms.binWidth*bins[i]^2 for i in 1:length(bins)]
    rdfNorm = ones(Float64, systemParms.Nbins)
    for i in 1:length(rdfNorm)
        rdfNorm[i] = systemParms.V/Npairs /shellVolumes[i]
    end
    hist .*= rdfNorm
    return(hist)
end

"""
function atomicEnergy(inputlayer, model)

Computes the potential energy of one particle
from the input layer of the neural network
"""
function atomicEnergy(inputlayer, model)
    E::Float64 = model(inputlayer)[1]
    return(E)
end

"""
function totalEnergy(symmFuncMatrix, model)

Computes the total potential energy of the system
"""
function totalEnergyScalar(symmFuncMatrix, model)
    N = size(symmFuncMatrix)[1]
    E = 0.
    for i in 1:N
        E += atomicEnergy(symmFuncMatrix[i, :], model)
    end
    return(E)
end

function getIndexesForUpdating(distanceVector2, systemParms, NNParms)
    indexes = []
    for i in 1:systemParms.N
        if distanceVector2[i] < NNParms.maxR
            append!(indexes, i)
        end
    end
    return(indexes)
end

function totalEnergyVector(symmFuncMatrix, model, indexesForUpdate, previousE)
    N = size(symmFuncMatrix)[1]
    E = copy(previousE)
    for i in indexesForUpdate
        E[i] = atomicEnergy(symmFuncMatrix[i, :], model)
    end
    return(E)
end

function totalEnergyVectorInit(symmFuncMatrix, model)
    N = size(symmFuncMatrix)[1]
    E = Array{Float64}(undef, N)
    for i in 1:N
        E[i] = atomicEnergy(symmFuncMatrix[i, :], model)
    end
    return(E)
end

"""
function mcmove!(mcarrays, E, model, NNParms, systemParms, rng)

Performs a Metropolis Monte Carlo
displacement move using a neural network
to predict energies from the symmetry function matrix
"""
function mcmove!(mcarrays, E, E_previous_vector, model, NNParms, systemParms, rng)
    # Unpack mcarrays
    frame, distanceMatrix, G2Matrix1 = mcarrays

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(systemParms.N))

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Take a copy of the previous energy value
    E1 = copy(E)
    
    # Displace the particle
    dr = [systemParms.Δ*(rand(rng, Float64) - 0.5), 
          systemParms.Δ*(rand(rng, Float64) - 0.5), 
          systemParms.Δ*(rand(rng, Float64) - 0.5)]

    positions(frame)[:, pointIndex] .+= dr

    # Compute the updated distance vector
    distanceVector2 = Array{Float64}(undef, systemParms.N)
    distanceVector2 = updatedistance!(frame, distanceVector2, pointIndex)

    # Acceptance counter
    accepted = 0


    # Reject the move prematurely if a single pair distance
    # is below the repulsion limit
    for distance in distanceVector2
        if distance < systemParms.repulsionLimit && distance > 0.
            # Revert to the previous configuration
            positions(frame)[:, pointIndex] .-= dr
            # Pack mcarrays
            mcarrays = (frame, distanceMatrix, G2Matrix1)
            # Finish function execution
            return(mcarrays, E, E_previous_vector, accepted)
        end
    end

    indexesForUpdate = getIndexesForUpdating(distanceVector2, systemParms, NNParms)
    
    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(G2Matrix1)
    updateG2Matrix!(G2Matrix2, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

    # Compute the energy again
    # E2 = totalEnergyScalar(G2Matrix2, model) 
    newEnergyVector = totalEnergyVector(G2Matrix2, model, indexesForUpdate, E_previous_vector)
    E2 = sum(newEnergyVector)
    
    # Get energy difference
    ΔE = E2 - E1
    
    # Accept or reject the move
    if rand(rng, Float64) < exp(-ΔE*systemParms.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector2
        distanceMatrix[:, pointIndex] = distanceVector2
        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix2)
        return(mcarrays, E, newEnergyVector, accepted)
    else
        positions(frame)[:, pointIndex] .-= dr
        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix1)
        return(mcarrays, E, E_previous_vector, accepted)
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
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Take a random frame from the equilibrated trajectory
    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1
    frameId = rand(rng_xor, 1:nframes) # Don't take the first frame
    frame = deepcopy(read_step(traj, frameId))

    # Start writing MC trajectory
    writetraj(positions(frame), systemParms, trajFile, 'w')
    writetraj(positions(frame), systemParms, pdbFile, 'w')

    # Get the number of data points
    totalDataPoints = Int(MCParms.steps / MCParms.outfreq)
    prodDataPoints = Int((MCParms.steps - MCParms.Eqsteps) / MCParms.outfreq)

    # Build the distance matrix
    distanceMatrix = builddistanceMatrix(frame)

    # Build the G2 matrix
    G2Matrix = buildG2Matrix(distanceMatrix, NNParms)
    
    # Prepare a tuple of arrays that change duing the mcmove!
    mcarrays = (frame, distanceMatrix, G2Matrix)

    # Initialize the distance histogram accumulator
    histAccumulator = zeros(Float64, systemParms.Nbins)

    # Build the cross correlation arrays for training,
    # an additional distance histogram array
    # and the G2 matrix accumulator
    if globalParms.mode == "training"
        hist = zeros(Float64, systemParms.Nbins)
        G2MatrixAccumulator = zeros(size(G2Matrix))
        crossAccumulators = crossAccumulatorsInit(NNParms, systemParms)
    end

    # Initialize the starting energy and the energy array
    E_previous_vector = totalEnergyVectorInit(G2Matrix, model)
    E = sum(E_previous_vector)
    energies = zeros(totalDataPoints + 1)
    energies[1] = E

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @inbounds @fastmath for step in 1:MCParms.steps
        mcarrays, E, E_previous_vector, accepted = mcmove!(mcarrays, E, E_previous_vector, model, NNParms, systemParms, rng_xor)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        # Perform MC step adjustment during the equilibration
        if MCParms.stepAdjustFreq > 0 && step % MCParms.stepAdjustFreq == 0 && step < MCParms.Eqsteps
            stepAdjustment!(systemParms, MCParms, acceptedIntermediate)
            acceptedIntermediate = 0
        end

        # Collect the output energies
        if step % MCParms.outfreq == 0
            energies[Int(step/MCParms.outfreq) + 1] = E
        end

         # MC trajectory output
         if step % MCParms.trajout == 0
            writetraj(positions(mcarrays[1]), systemParms, trajFile, 'a')
        end

        # Accumulate the distance histogram
        if step % MCParms.outfreq == 0 && step > MCParms.Eqsteps
            frame, distanceMatrix, G2Matrix = mcarrays
            # Update the cross correlation array during the training
            if globalParms.mode == "training"
                hist = hist!(distanceMatrix, hist, systemParms)
                histAccumulator .+= hist
                G2MatrixAccumulator .+= G2Matrix
                # Normalize the histogram to RDF
                normalizehist!(hist, systemParms)
                updateCrossAccumulators!(crossAccumulators, G2Matrix, hist, model)
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
    frame, distanceMatrix, G2Matrix = mcarrays
    if globalParms.mode == "training"
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
        G2MatrixAccumulator ./= prodDataPoints
    end

    # Normalize the sampled distance histogram
    histAccumulator ./= prodDataPoints
    normalizehist!(histAccumulator, systemParms)
    
    if globalParms.mode == "training"
        MCoutput = MCAverages(histAccumulator, energies, crossAccumulators, G2MatrixAccumulator, acceptanceRatio, systemParms)
        return(MCoutput)
    else
        MCoutput = MCAverages(histAccumulator, energies, nothing, nothing, acceptanceRatio, systemParms)
        return(MCoutput)
    end
end

"""
function stepAdjustment!(systemParms, MCParms, acceptedIntermediate)

MC step length adjustment
"""
function stepAdjustment!(systemParms, MCParms, acceptedIntermediate)
    acceptanceRatio = acceptedIntermediate / MCParms.stepAdjustFreq
    systemParms.Δ = acceptanceRatio * systemParms.Δ / systemParms.targetAR
    return(systemParms)
end