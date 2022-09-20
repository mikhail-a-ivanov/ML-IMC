using StaticArrays
using LinearAlgebra
using Flux
using BSON: @save, @load

struct MCInput
    parameters
    systemParms
    model
end

struct MCTrainingOutputs
    descriptor
    energies
    crossAccumulators
    acceptanceRatio
    systemParms
end

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
function crossAccumulatorsInit(parameters, systemParms)

Initialize cross correlation accumulator arrays
"""
function crossAccumulatorsInit(parameters, systemParms)
    crossAccumulators = []
    nlayers = length(parameters.neurons)
    for layerId in 2:nlayers
        if layerId < nlayers
            append!(crossAccumulators, [zeros(Float32, (systemParms.Nbins, 
                    parameters.neurons[layerId - 1] * parameters.neurons[layerId]))])
            append!(crossAccumulators, [zeros(Float32, (systemParms.Nbins, 
                    parameters.neurons[layerId]))])
        else
            append!(crossAccumulators, [zeros(Float32, (systemParms.Nbins, 
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
function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)

Computes the gradients of the descriptor with respect to the network parameters
"""
function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)
    descriptorGradients = []
    for (accumulator, ensemble) in zip(crossAccumulators, ensembleCorrelations)
        gradients = -Float32(systemParms.β) .* (accumulator - ensemble)
        append!(descriptorGradients, [gradients])
    end
    return(descriptorGradients)
end

function computeLossGradients(crossAccumulators, descriptorNN, descriptorref, model, systemParms)
    lossGradients = []
    ensembleCorrelations = computeEnsembleCorrelation(descriptorNN, model)
    descriptorGradients = computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)
    # Compute derivative of loss with respect to the descriptor
    dLdS = zeros(Float32, length(descriptorNN))
    for i in 1:length(dLdS)
        dLdS[i] = 2*(descriptorNN[i] - descriptorref[i])
    end
    for (gradient, parameters) in zip(descriptorGradients, Flux.params(model))
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
    for (gradient, parameters) in zip(lossGradients, Flux.params(model))
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
function modelInit(parameters)

Generates a neural network with zero weigths
in the first layer and random values in other layers
"""
function modelInit(parameters)
    # Build initial model
    network = buildNetwork!(parameters)
    println("Building a model...")
    model = buildchain(network...)
    println(model)
    println("   Number of layers: $(length(parameters.neurons)) ")
    println("   Number of neurons in each layer: $(parameters.neurons)")
    nlayers = length(model.layers)
    # Initialize weights
    for (layerId, layer) in enumerate(model.layers)
        for column in eachrow(layer.weight)
            if layerId == 1
                Flux.Optimise.update!(column, column)
            end
        end
    end
    return(model)
end

"""
function optInit(parameters)

Initializes the optimizer
"""
function optInit(parameters)
    if parameters.optimizer == "Momentum"
        opt = Momentum(parameters.rate, parameters.momentum)
    elseif parameters.optimizer == "Descent"
        opt = Descent(parameters.rate)
    else
        opt = Descent(parameters.rate)
        println("Other types of optimizers are currently not supported!")
    end
    return(opt)
end

"""
function collectAverages(name, outputs, outputIDs)

USE THIS: getfield(structName, fieldnames(struct))!!!

Collects averages of a given name from pmap outputs
"""
function collectAverages(propertyName, outputs, outputIDs)
    averages = []
    for outputID in outputIDs
        append!(averages, [outputs[outputID].propertyName])
    end
    return(mean(average for average in averages))
end

"""
function train!(parameters, systemParmsList, model, opt, refRDFs)

Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) training iterations
"""
function train!(parameters, systemParmsList, model, opt, refRDFs)
    # Initialize the list of loss values
    losses = []
    # Run training iterations
    iteration = 1
    ### REMOVE LATER
    descriptorref = refRDFs[1]
    systemParms = systemParmsList[1] # I will check the new parameters structs only
    ###
    while iteration <= parameters.iters
        iterString = lpad(iteration, 2, '0')
        println("Iteration $(iteration)...")
        # Prepare multireference input
        nsystems = length(systemParmsList)
        multiReferenceInput = []
        for systemId in 1:nsystems
            input = MCInput(parameters, systemParmsList[systemId], model)
            append!(multiReferenceInput, [input])
        end 
        nsets = Int(nworkers()/nsystems)
        inputs = []
        for setId in 1:nsets
            append!(inputs, multiReferenceInput)
        end
     
        # Run the simulation in parallel
        outputs = pmap(mcsample!, inputs)

        NNRDFs = []
        energies = []
        meanAcceptanceRatios = []
        crossAccumulators = []

        for systemId in 1:nsystems
            systemParms = systemParmsList[systemId]
            println("System $(systemParms.systemName):")
            # Find the corresponding outputs
            outputIDs = []
            for outputID in eachindex(outputs)
                if systemParms.systemName == outputs[outputID].systemParms
                    append!(outputIDs, outputID)
                end
            end
            NNRDF = collectAverages(descriptor, outputs, outputIDs)
            append!(NNRDFs, [NNRDF])
            meanSystemEnergy = collectAverages(energies, outputs, outputIDs)
            append!(energies, [meanSystemEnergy])
            meanAcceptanceRatio = collectAverages(acceptanceRatio, outputs, outputIDs)
            println("Mean acceptance ratio = ", round(meanAcceptanceRatio, digits=4))
            append!(meanAcceptanceRatios, meanAcceptanceRatio)
            crossAccumulator = collectAverages(crossAccumulators, outputs, outputIDs)
            append!(crossAccumulators, crossAccumulator)
        end

        # Compute average loss
        lossvalue = 0
        for systemId in 1:nsystems    
            lossvalue += loss(NNRDFs[systemId], refRDFs[systemId])
        end
        lossvalue /= nsystems
        append!(losses, lossvalue)

        # Compute the gradients and update the model
        lossGradients = []

        # Write the descriptor and compute the gradients
        for systemId in 1:nsystems
            systemParms = systemParmsList[systemId]
            name = systemParms.systemName
            writedescriptor("RDFNN-$(name)-iter-$(iterString).dat", NNRDFs[systemId], systemParms)
            lossGradient = computeLossGradients(crossAccumulators[systemId], 
                                                NNRDFs[systemId], refRDFs[systemId], model, systemParms)
            append!(lossGradients, [lossGradient])

            # Write averaged energies
            writeenergies("energies-$(name)-iter-$(iterString).dat", energies[systemId], parameters, 10)
        end
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])

        # Write the model (before training!) and the gradients
        @save "model-iter-$(iterString).bson" model
        @save "gradients-iter-$(iterString).bson" meanLossGradients

        # Update the model if the loss decreased
        updatemodel!(model, opt, meanLossGradients)
        # Move on to the next iteration
        iteration += 1
    end
    println("The training is finished!")
    return
end