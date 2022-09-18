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
function histpart!(distanceVector, hist, binWidth)

Accumulates pair distances from a distance vector
(one particle) to a histogram
"""
function histpart!(distanceVector, hist, binWidth)
    N = length(distanceVector)
    @inbounds @fastmath for i in 1:N
        if distanceVector[i] != 0
            histIndex = floor(Int32, 0.5 + distanceVector[i]/binWidth)
            if histIndex <= length(hist)
                hist[histIndex] += 1
            end
        end
    end
    return(hist)
end

"""
normalizehist!(hist, parameters)

Normalizes one particle distance histogram to RDF
"""
function normalizehist!(hist, parameters)
    bins = [bin*parameters.binWidth for bin in 1:parameters.Nbins]
    shellVolumes = [4*π*parameters.binWidth*bins[i]^2 for i in 1:length(bins)]
    rdfNorm = ones(Float32, parameters.Nbins)
    for i in 2:length(rdfNorm)
        rdfNorm[i] = parameters.V/parameters.N * 1/shellVolumes[i]
    end
    hist .*= rdfNorm
    return(hist)
end

"""
neuralenergy(inputlayer, model)

Computes the potential energy of one particle
from the input layer of the neural network
"""
function neuralenergy(inputlayer, model)
    E::Float64 = model(inputlayer)[1]
    return(E)
end

"""
mcmove!(mcarrays, E, model, parameters, step, rng)

Performs a Metropolis Monte Carlo
displacement move using a neural network
to predict energies from pair correlation functions (pairdescriptor)
"""
function mcmove!(mcarrays, E, model, parameters, step, rng)
    # Acceptance counter
    accepted = 0

    # Unpack mcarrays
    if parameters.mode == "training"
        frame, distanceMatrix, pairdescriptorNN, crossAccumulators = mcarrays
    else
        frame, distanceMatrix, pairdescriptorNN = mcarrays
    end

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(length(frame)))
    
    # Allocate the distance vector
    distanceVector = distanceMatrix[:, pointIndex]

    # Allocate and compute the pair descriptor
    pairdescriptor1 = zeros(Float32, parameters.Nbins)
    histpart!(distanceVector, pairdescriptor1, parameters.binWidth)
    normalizehist!(pairdescriptor1, parameters)
    
    # Compute the energy
    E1 = neuralenergy(pairdescriptor1, model)
    
    # Displace the particle
    dr = [parameters.delta*(rand(rng, Float64) - 0.5), 
          parameters.delta*(rand(rng, Float64) - 0.5), 
          parameters.delta*(rand(rng, Float64) - 0.5)]

    
    positions(frame)[:, pointIndex] += dr

    # Update distance
    updatedistance!(frame, distanceVector, pointIndex)

    # Reject the move prematurely if a single pair distance
    # is below the repulsion limit
    if parameters.repulsionLimit > 0
        for distance in distanceVector
            if distance < parameters.repulsionLimit && distance > 0.
                # Revert to the previous configuration
                positions(frame)[:, pointIndex] -= dr
                # Update the descriptor data
                if step % parameters.outfreq == 0 && step > parameters.Eqsteps
                    for i in 1:parameters.Nbins
                        pairdescriptorNN[i] += pairdescriptor1[i] 
                    end
                    if parameters.mode == "training"
                        # Update cross correlation accumulators
                        updateCrossAccumulators!(crossAccumulators, pairdescriptor1, model)
                    end
                end
                # Pack mcarrays
                if parameters.mode == "training"
                    mcarrays = (frame, distanceMatrix, pairdescriptorNN, crossAccumulators)
                else
                    mcarrays = (frame, distanceMatrix, pairdescriptorNN)
                end
                # Finish function execution
                return(mcarrays, E, accepted)
            end
        end
    end
    
    # Compute the new descriptor
    pairdescriptor2 = zeros(Float32, parameters.Nbins)
    histpart!(distanceVector, pairdescriptor2, parameters.binWidth)
    normalizehist!(pairdescriptor2, parameters)
    
    # Compute the energy again
    E2 = neuralenergy(pairdescriptor2, model)
    
    # Get energy difference
    ΔE = E2 - E1
    
    if rand(rng, Float64) < exp(-ΔE*parameters.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector
        distanceMatrix[:, pointIndex] = distanceVector
        # Update the descriptor data
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                pairdescriptorNN[i] += pairdescriptor2[i] 
            end
            if parameters.mode == "training"
                # Update cross correlation accumulators
                updateCrossAccumulators!(crossAccumulators, pairdescriptor2, model)
            end
        end
    else
        positions(frame)[:, pointIndex] -= dr
        # Update the descriptor data
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                pairdescriptorNN[i] += pairdescriptor1[i] 
            end
            if parameters.mode == "training"
                # Update cross correlation accumulators
                updateCrossAccumulators!(crossAccumulators, pairdescriptor1, model)
            end
        end
    end
    # Pack mcarrays
    if parameters.mode == "training"
        mcarrays = (frame, distanceMatrix, pairdescriptorNN, crossAccumulators)
    else
        mcarrays = (frame, distanceMatrix, pairdescriptorNN)
    end
    return(mcarrays, E, accepted)
end

"""
mcsample!(input)
(input = parameters, model)
Runs the Monte Carlo simulation for a given
input configuration, set of parameters
and the neural network model
"""
function mcsample!(input)
    # Initialize RNG
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Unpack the inputs
    parameters, model = input

    # Get the number of data points
    totalDataPoints = Int(parameters.steps / parameters.outfreq)
    prodDataPoints = Int((parameters.steps - parameters.Eqsteps) / parameters.outfreq)

    # Take a random frame from the equilibrated trajectory
    traj = Trajectory(parameters.trajfile)
    nframes = Int(size(traj)) - 1
    frameId = rand(rng_xor, Int(nframes/2):nframes) # Take frames from the second half
    frame = deepcopy(read_step(traj, frameId))

    # Build the distance matrix
    distanceMatrix = builddistanceMatrix(frame)

    # Initialize the energy
    E::Float64 = 0.

    # Allocate the pair correlation functions
    pairdescriptorNN = zeros(Float32, parameters.Nbins)

    # Prepare a tuple of arrays that change duing the mcmove!
    # and optionally build the cross correlation arrays
    if parameters.mode == "training"
        crossAccumulators = crossAccumulatorsInit(parameters)
        mcarrays = (frame, distanceMatrix, pairdescriptorNN, crossAccumulators)
    else
        mcarrays = (frame, distanceMatrix, pairdescriptorNN)
    end

    # Initialize the energy array
    energies = zeros(totalDataPoints)

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @inbounds @fastmath for step in 1:parameters.steps
        mcarrays, E, accepted = mcmove!(mcarrays, E, model, parameters, step, rng_xor)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        if step % parameters.outfreq == 0
            energies[Int(step/parameters.outfreq)] = E
        end

        # Perform MC step adjustment during the equilibration
        if parameters.stepAdjustFreq > 0 && step % parameters.stepAdjustFreq == 0 && step < parameters.Eqsteps
            stepAdjustment!(parameters, acceptedIntermediate)
            acceptedIntermediate = 0
        end
    end
    acceptanceRatio = acceptedTotal / parameters.steps

    # Unpack mcarrays and optionally normalize crossAccumulators
    if parameters.mode == "training"
        frame, distanceMatrix, pairdescriptorNN, crossAccumulators = mcarrays
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
    else
        frame, distanceMatrix, pairdescriptorNN = mcarrays
    end

    # Normalize the pair correlation functions
    pairdescriptorNN ./= prodDataPoints

    println("Max displacement = ", round(parameters.delta, digits=4))
    println("Acceptance ratio = ", round(acceptanceRatio, digits=4))

    if parameters.mode == "training"
        return(pairdescriptorNN, energies, crossAccumulators, acceptanceRatio)
    else
        return(pairdescriptorNN, energies, acceptanceRatio)
    end
end

"""
function stepAdjustment!(parameters, acceptedIntermediate)

MC step length adjustment
"""
function stepAdjustment!(parameters, acceptedIntermediate)
    acceptanceRatio = acceptedIntermediate / parameters.stepAdjustFreq
    parameters.delta = acceptanceRatio * parameters.delta / parameters.targetAR
    return(parameters)
end