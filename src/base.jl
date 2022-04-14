using Printf
using RandomNumbers
using Statistics
using StaticArrays
using LinearAlgebra
using Flux

include("distances.jl")
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
    V = parameters.box[1] * parameters.box[2] * parameters.box[3]
    #Npairs::Int = parameters.N * (parameters.N - 1) / 2
    bins = [bin*parameters.binWidth for bin in 1:parameters.Nbins]
    shellVolumes = [4*π*parameters.binWidth*bins[i]^2 for i in 1:length(bins)]
    rdfNorm = ones(Float32, parameters.Nbins)
    for i in 2:length(rdfNorm)
        rdfNorm[i] = V/parameters.N * 1/shellVolumes[i]
    end
    hist .*= rdfNorm
    return(hist)
end

"""
neuralenergy(inputlayer, model, parameters)

Computes the potential energy of one particle
from the input layer of the neural network
"""
function neuralenergy(inputlayer, model)
    E::Float64 = model(inputlayer)[1]
    return(E)
end

"""
mcmove!(conf, E, step, parameters, model, pairdescriptorNN, rng)

Performs a Metropolis Monte Carlo
displacement move using a neural network
to predict energies from pair correlation functions (pairdescriptor)
"""
function mcmove!(conf, distanceMatrix, crossWeights, crossBiases, E, step, parameters, model, pairdescriptorNN, rng)
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
    dr = SVector{3, Float64}(parameters.delta*(rand(rng, Float64) - 0.5), 
                             parameters.delta*(rand(rng, Float64) - 0.5), 
                             parameters.delta*(rand(rng, Float64) - 0.5))
    
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
    deltaE = E2 - E1
    # Acceptance counter
    accepted = 0
    
    if rand(rng, Float64) < exp(-deltaE*parameters.beta)
        accepted += 1
        E += deltaE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector
        distanceMatrix[:, pointIndex] = distanceVector
        # Update the descriptor data
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                pairdescriptorNN[i] += pairdescriptor2[i] 
            end
            # Update cross correlation arrays
            crossCorrelation!(pairdescriptor2, model, crossWeights, crossBiases)
        end
    else
        conf[pointIndex] -= dr
        # Update the descriptor data
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                pairdescriptorNN[i] += pairdescriptor1[i] 
            end
            # Update cross correlation arrays
            crossCorrelation!(pairdescriptor1, model, crossWeights, crossBiases)
        end
    end
    return(conf, distanceMatrix, crossWeights, crossBiases, E, pairdescriptorNN, accepted)
end

"""
mcrun!(input)
(input = conf, parameters, model)
Runs the Monte Carlo simulation for a given
input configuration, set of parameters
and the neural network model
"""
function mcrun!(input)
    # Unpack the inputs
    conf, parameters, model = input
    TotalDataPoints = Int(parameters.steps / parameters.outfreq)
    prodDataPoints = Int((parameters.steps - parameters.Eqsteps) / parameters.outfreq)

    # Allocate and initialize the energy
    energies = zeros(TotalDataPoints + 1)
    E = 0.

    # Allocate the pair correlation functions
    pairdescriptorNN = zeros(Float32, parameters.Nbins)

    # Initialize RNG
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Build the distance matrix
    distanceMatrix = builddistanceMatrix(conf, parameters.box)

    # Build the cross correlation arrays
    crossWeights = zeros(Float32, parameters.Nbins, length(model.weight))
    crossBiases = zeros(Float32, parameters.Nbins)

    # Acceptance counter
    acceptedTotal = 0

    # Run MC simulation
    @inbounds @fastmath for step in 1:parameters.steps
        conf, distanceMatrix, crossWeights, crossBiases, E, pairdescriptorNN, accepted = 
            mcmove!(conf, distanceMatrix, crossWeights, crossBiases, E, step, 
            parameters, model, pairdescriptorNN, rng_xor)
        acceptedTotal += accepted
        if step % parameters.outfreq == 0
            energies[Int(step/parameters.outfreq) + 1] = E
        end
    end
    acceptanceRatio = acceptedTotal / parameters.steps

    # Normalize the pair correlation functions
    pairdescriptorNN ./= Float32(prodDataPoints)

    # Normalize the cross correlation arrays 
    crossWeights ./= prodDataPoints
    crossBiases ./= prodDataPoints

    println("Acceptance ratio = $(acceptanceRatio)")
    return(pairdescriptorNN, energies, crossWeights, crossBiases, acceptanceRatio)
end

"""
crossCorrelation!(descriptor, model, crossWeights, crossBiases)

Updated cross correlation arrays
"""
function crossCorrelation!(descriptor, model, crossWeights, crossBiases)
    dHdw, dHdb = gradient(neuralenergy, descriptor, model)[2]
    crossWeights .+= (descriptor * dHdw) # Matrix Nbins x Nweights
    crossBiases .+= (descriptor .* dHdb) # Vector Nbins
    return(crossWeights, crossBiases)
end

"""
function computeDerivatives(crossWeights, crossBiases, descriptor, model, parameters)

Compute dL/dl derivatives for training (based on IMC method)
"""
function computeDerivatives(crossWeights, crossBiases, descriptorNN, descriptorref, model, parameters)
    # Compute <dH/dl> * <S>
    dHdw, dHdb = gradient(neuralenergy, descriptorNN, model)[2]
    productWeights = Float32.(descriptorNN * dHdw)
    productBiases = Float32.(descriptorNN .* dHdb)

    # Compute descriptor gradients
    dSdw = -Float32(parameters.beta) .* (crossWeights - productWeights)
    dSdb = -Float32(parameters.beta) .* (crossBiases - productBiases)

    # Loss derivative
    dL = lossDerivative(descriptorNN, descriptorref)

    # Loss total gradient
    dLdw = zeros(length(model.weight))
    for i in 1:length(dLdw)
        dLdw[i] = 2 * dL[i] * sum(dSdw[i, :]) # take a column from dS/dw matrix
    end
    dLdb = sum(dL .* dSdb)
    return(dLdw, dLdb)
end

"""
function updatemodel!(model, opt, dLdw, dLdb)

Update model parameters using the calculated 
gradients and the selected optimizer
"""
function updatemodel!(model, opt, dLdw, dLdb)
    Flux.Optimise.update!(opt, model.weight, dLdw)
    Flux.Optimise.update!(opt, model.bias, [dLdb])
end

"""
function loss(descriptorNN, descriptorref)

Compute the error function
"""
function loss(descriptorNN, descriptorref)
    loss = zeros(length(descriptorNN))
    for i in 1:length(loss)
        loss[i] = (descriptorNN[i] - descriptorref[i])^2
    end
    totalLoss = sum(loss)
    println("Loss = ", round(totalLoss, digits=8))
    return(totalLoss)
end

"""
function lossDerivative(descriptorNN, descriptorref)

Compute the loss derivative function for dL/dl calculation
"""
function lossDerivative(descriptorNN, descriptorref)
    loss = zeros(length(descriptorNN))
    for i in 1:length(loss)
        loss[i] = 2*(descriptorNN[i] - descriptorref[i])
    end
    return(loss)
end

"""
function mindistance(descriptor, parameters)

Returns the minimal occuring distance in the descriptor
"""
function mindistance(descriptor, parameters)
    for i in 1:parameters.Nbins
        if descriptor[i] != 0
            return((i - 1) * parameters.binWidth)
        end
    end
end

"""
function repulsion(descriptor, parameters, shift=0.01, stiffness=5)

Returns repulsion weights for the neural network
Functional form for repulsion: stiffness*[exp(-alpha*r)-shift]
alpha is a coefficient that makes sure that the repulsion term
goes to zero at minimal distance from the given pair correlation function (descriptor)
"""
function repulsion(descriptor, parameters, shift=0.01, stiffness=5)
    bins = [bin*parameters.binWidth for bin in 1:parameters.Nbins]
    minDistance = mindistance(descriptor, parameters)
    # Find alpha so that [exp(-alpha*r) - shift] goes to zero at minDistance
    alpha = -log(shift)/minDistance
    potential = zeros(Float32, parameters.Nbins)
    for i in 1:parameters.Nbins
        if bins[i] < minDistance
            potential[i] = stiffness*(exp(-alpha*bins[i])-shift)
        end
    end
    return(potential)
end

"""
function modelinit(descriptor, parameters, shift=0.01, stiffness=5)

Initializes the model with repulsion term
"""
function modelinit(descriptor, parameters, shift=0.01, stiffness=5)
    if parameters.activation == "identity"
        model = Dense(length(descriptor), 1, identity, bias=true)
    elseif parameters.activation == "tanh"
        model = Dense(length(descriptor), 1, tanh, bias=true)
    else
        println("Other types of activations are not supported.")
    end
    # Nullify weights (subtract weights from themselves)
    Flux.Optimise.update!(model.weight, model.weight)
    # Add repulsion term
    Flux.Optimise.update!(model.weight, -repulsion(descriptor, parameters, shift, stiffness)')
    return(model)
end