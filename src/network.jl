using StaticArrays
using LinearAlgebra
using Flux

"""
function energyGradients(descriptor, model)

Computes all gradients of energy with respect
to all parameters in the given network
Assumes that the last layer does not contain bias parameters
"""
function computeEnergyGradients(descriptor, model)
    energyGradients = []
    # Compute energy gradients
    gs = gradient(neuralenergy, descriptor, model)
    # Loop over the gradients and collect them in the array
    nlayers = length(model)
    # Structure: gs[2][1][layerId][1 - weigths; 2 - biases]
    for (layerId, layerGradients) in enumerate(gs[2][1]) 
        if layerId != nlayers
            weightGradients = layerGradients[1]
            append!(energyGradients, [weightGradients])
            biasGradients = layerGradients[2]
            append!(energyGradients, [biasGradients])
        else
            weightGradients = layerGradients[1]
            append!(energyGradients, [weightGradients])
        end
    end
    return(energyGradients)
end

"""
function computeCrossCorrelation(descriptor, energyGradients)

Computes cross products of the descriptor and energy gradients
"""
function computeCrossCorrelation(descriptor, energyGradients)
    crossCorrelations = []
    for gradient in energyGradients
        cross = descriptor * gradient[:]' # Matrix Nbins x Nparameters
        append!(crossCorrelations, [cross])
    end
    return(crossCorrelations)
end

"""
function crossAccumulatorsInit(parameters)

Initialize cross correlation accumulator arrays
"""
function crossAccumulatorsInit(parameters)
    crossAccumulators = []
    nlayers = length(parameters.neurons)
    for layerId in 2:nlayers
        if layerId < nlayers
            append!(crossAccumulators, [zeros(Float32, (parameters.Nbins, 
                    parameters.neurons[layerId - 1] * parameters.neurons[layerId]))])
            append!(crossAccumulators, [zeros(Float32, (parameters.Nbins, 
                    parameters.neurons[layerId]))])
        else
            append!(crossAccumulators, [zeros(Float32, (parameters.Nbins, 
                    parameters.neurons[layerId - 1] * parameters.neurons[layerId]))])
        end
    end
    return(crossAccumulators)
end

"""
function updateCrossAccumulators(crossAccumulators, descriptor, model)

Updates cross accumulators by performing element-wise summation
of the cross accumulators with the new cross correlation data
"""
function updateCrossAccumulators!(crossAccumulators, descriptor, model)
    energyGradients = computeEnergyGradients(descriptor, model)
    newCrossCorrelations = computeCrossCorrelation(descriptor, energyGradients)
    for (cross, newCross) in zip(crossAccumulators, newCrossCorrelations)
        cross .+= newCross
    end
    return(crossAccumulators)
end

"""
function computeEnsembleCorrelation(descriptor, model)

Computes correlations of the ensemble averages of the descriptor
and the energy gradients
"""
function computeEnsembleCorrelation(descriptor, model)
    energyGradients = computeEnergyGradients(descriptor, model)
    ensembleCorrelations = computeCrossCorrelation(descriptor, energyGradients)
    return(ensembleCorrelations)
end

"""
function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, parameters)

Computes the gradients of the descriptor with respect to the network parameters
"""
function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, parameters)
    descriptorGradients = []
    for (accumulator, ensemble) in zip(crossAccumulators, ensembleCorrelations)
        gradients = -Float32(parameters.beta) .* (accumulator - ensemble)
        append!(descriptorGradients, [gradients])
    end
    return(descriptorGradients)
end

function computeLossGradients(crossAccumulators, descriptorNN, descriptorref, model, parameters)
    lossGradients = []
    ensembleCorrelations = computeEnsembleCorrelation(descriptorNN, model)
    descriptorGradients = computeDescriptorGradients(crossAccumulators, ensembleCorrelations, parameters)
    # Compute derivative of loss with respect to the descriptor
    dLdS = zeros(Float32, length(descriptorNN))
    for i in 1:length(dLdS)
        dLdS[i] = 2*(descriptorNN[i] - descriptorref[i])
    end
    for (gradient, parameters) in zip(descriptorGradients, params(model))
        lossGradient = dLdS' * gradient
        lossGradient = reshape(lossGradient, size(parameters))
        append!(lossGradients, [lossGradient])
    end
    return(lossGradients)
end

"""
function updatemodel!(model, opt, lossGradients)

Updates the network parameters
"""
function updatemodel!(model, opt, lossGradients)
    for (gradient, parameters) in zip(lossGradients, params(model))
        Flux.Optimise.update!(opt, parameters, gradient)
    end
    return
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
function buildNetwork!(parameters)

Combines input arguments for neural network building
Note: updates parameters.neurons
"""
function buildNetwork!(parameters)
    if parameters.neurons == [0]
        parameters.neurons = []
    end
    # Add input and output layers to the parameters.neurons
    pushfirst!(parameters.neurons, parameters.Nbins)
    push!(parameters.neurons, 1)
    nlayers = length(parameters.neurons)
    network = []
    for layerId in 2:nlayers
        if layerId < nlayers
        append!(network, [(parameters.neurons[layerId - 1], parameters.neurons[layerId],
                getfield(Main, Symbol(parameters.activation)))])
        else
            append!(network, [(parameters.neurons[layerId - 1], parameters.neurons[layerId])])
        end
    end
    return(network)
end

"""
function buildchain(args...)

Build a multilayered neural network
"""
function buildchain(args...)
    nlayers = length(args)
    layers = []
    for (layerId, arg) in enumerate(args)
        if layerId < nlayers
            layer = Dense(arg...)
        else
            layer = Dense(arg..., bias=false)
        end
        append!(layers, [layer])
    end
    model = Chain(layers...)
    return(model)
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

Generates a neural network with or without repulsion term in the input layer.
If parameters.paramsInit is set to repulsion then the repulsion terms are applied 
for each set of weights associated with a single neuron in the next layer.
Weights in all the other layers are set to unity.
Otherwise all the weights are set to random and biases to zero
"""
function modelinit(descriptor, parameters, shift=0.01, stiffness=5)
    # Build initial model
    network = buildNetwork!(parameters)
    println("Building a model...")
    model = buildchain(network...)
    println(model)
    println("   Number of layers: $(length(parameters.neurons)) ")
    println("   Number of neurons in each layer: $(parameters.neurons)")
    println("   Parameter initialization: $(parameters.paramsInit)")
    if parameters.paramsInit == "repulsion"
        nlayers = length(model.layers)
        # Initialize weights
        for (layerId, layer) in enumerate(model.layers)
            for column in eachrow(layer.weight)
                Flux.Optimise.update!(column, column)
                if layerId == 1
                    Flux.Optimise.update!(column, -repulsion(descriptor, parameters, shift, stiffness))
                elseif layerId < nlayers
                    Flux.Optimise.update!(column, -ones(Float32, length(column)))
                else
                    # Multiply the weights by the fraction of input neurons and second-to-last neurons
                    # Migth be useful for many-layered networks, multiplier of unity is ok for one hidden layer 
                    #weightMultiplier = network[1][1] / network[end][1]  
                    weightMultiplier = 1
                    Flux.Optimise.update!(column, -weightMultiplier * ones(Float32, length(column)))
                end
            end
        end
    end
    return(model)
end