using Printf
using RandomNumbers
using Statistics
using StaticArrays
using LinearAlgebra
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
    # Unpack mcarrays
    if parameters.mode == "training"
        conf, distanceMatrix, pairdescriptorNN, crossAccumulators = mcarrays
    else
        conf, distanceMatrix, pairdescriptorNN = mcarrays
    end

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(length(conf)))
    
    # Allocate the distance vector
    distanceVector = distanceMatrix[:, pointIndex]

    # Allocate and compute the pair descriptor
    pairdescriptor1 = zeros(Float32, parameters.Nbins)
    histpart!(distanceVector, pairdescriptor1, parameters.binWidth)
    if parameters.paircorr == "RDF"
        normalizehist!(pairdescriptor1, parameters)
    end
    
    # Compute the energy
    E1 = neuralenergy(pairdescriptor1, model)
    
    # Displace the particle
    dr = SVector{3, Float32}(parameters.delta*(rand(rng, Float32) - 0.5), 
                             parameters.delta*(rand(rng, Float32) - 0.5), 
                             parameters.delta*(rand(rng, Float32) - 0.5))
    
    conf[pointIndex] += dr
    
    # Update distance and compute the new descriptor
    pairdescriptor2 = zeros(Float32, parameters.Nbins)
    updatedistance!(conf, parameters.box, distanceVector, pointIndex)
    histpart!(distanceVector, pairdescriptor2, parameters.binWidth)
    if parameters.paircorr == "RDF"
        normalizehist!(pairdescriptor2, parameters)
    end
    
    # Compute the energy again
    E2 = neuralenergy(pairdescriptor2, model)
    
    # Get energy difference
    ΔE = E2 - E1
    # Acceptance counter
    accepted = 0
    
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
        conf[pointIndex] -= dr
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
        mcarrays = (conf, distanceMatrix, pairdescriptorNN, crossAccumulators)
    else
        mcarrays = (conf, distanceMatrix, pairdescriptorNN)
    end
    return(mcarrays, E, accepted)
end

"""
mcsample!(input)
(input = conf, parameters, model)
Runs the Monte Carlo simulation for a given
input configuration, set of parameters
and the neural network model
"""
function mcsample!(input)
    # Initialize RNG
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Unpack the inputs
    conf, parameters, model = input

    # Get the number of data points
    totalDataPoints = Int(parameters.steps / parameters.outfreq)
    prodDataPoints = Int((parameters.steps - parameters.Eqsteps) / parameters.outfreq)

    # Build the distance matrix
    distanceMatrix = builddistanceMatrix(conf, parameters.box)

    # Initialize the energy
    E::Float64 = 0.

    # Allocate the pair correlation functions
    pairdescriptorNN = zeros(Float32, parameters.Nbins)

    # Prepare a tuple of arrays that change duing the mcmove!
    # and optionally build the cross correlation arrays
    if parameters.mode == "training"
        crossAccumulators = crossAccumulatorsInit(parameters)
        mcarrays = (conf, distanceMatrix, pairdescriptorNN, crossAccumulators)
    else
        mcarrays = (conf, distanceMatrix, pairdescriptorNN)
    end

    # Initialize the energy array
    energies = zeros(totalDataPoints)

    # Acceptance counter
    acceptedTotal = 0

    # Run MC simulation
    @inbounds @fastmath for step in 1:parameters.steps
        mcarrays, E, accepted = mcmove!(mcarrays, E, model, parameters, step, rng_xor)
        acceptedTotal += accepted
        if step % parameters.outfreq == 0
            energies[Int(step/parameters.outfreq)] = E
        end
    end
    acceptanceRatio = acceptedTotal / parameters.steps

    # Unpack mcarrays and optionally normalize crossAccumulators
    if parameters.mode == "training"
        conf, distanceMatrix, pairdescriptorNN, crossAccumulators = mcarrays
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
    else
        conf, distanceMatrix, pairdescriptorNN = mcarrays
    end

    # Normalize the pair correlation functions
    pairdescriptorNN ./= prodDataPoints

    println("Acceptance ratio = ", round(acceptanceRatio, digits=4))

    if parameters.mode == "training"
        return(pairdescriptorNN, energies, crossAccumulators, acceptanceRatio)
    else
        return(pairdescriptorNN, energies, acceptanceRatio)
    end
end

