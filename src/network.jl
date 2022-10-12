using StaticArrays
using LinearAlgebra
using Flux
using Distributed
using BSON: @save, @load

"""
struct MCSampleInput

Used for packaging mcsample! inputs
"""
struct MCSampleInput
    parameters
    systemParms
    model
end

"""
struct MCAverages

Used for packaging mcsample! outputs
"""
struct MCAverages
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
        weightGradients = layerGradients[1]
        append!(energyGradients, [weightGradients])
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
        append!(crossAccumulators, [zeros(Float32, (systemParms.Nbins, 
                parameters.neurons[layerId - 1] * parameters.neurons[layerId]))])
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

"""
function computeLossGradients(gradientInput)

Computes the final loss-network gradients
"""
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
    println("       Loss = ", round(totalLoss, digits=8))
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
        layer = Dense(arg..., bias=false)
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
function prepMCInputs(parameters, systemParmsList, model)

Prepares multi-reference inputs for mcsample! function
"""
function prepMCInputs(parameters, systemParmsList, model)
    nsystems = length(systemParmsList)
    multiReferenceInput = []
    for systemId in 1:nsystems
        input = MCSampleInput(parameters, systemParmsList[systemId], model)
        append!(multiReferenceInput, [input])
    end 
    nsets = Int(nworkers()/nsystems)
    inputs = []
    for setId in 1:nsets
        append!(inputs, multiReferenceInput)
    end
    return(inputs)
end

"""
function collectSystemAverages(outputs, refRDFs, systemParmsList, parameters, iterString)

Collects averages from different workers corresponding to one reference system
"""
function collectSystemAverages(outputs, refRDFs, systemParmsList, parameters, iterString)
    meanLoss = 0.
    systemOutputs = []
    for (systemId, systemParms) in enumerate(systemParmsList)
        println("   System $(systemParms.systemName):") 
        meanDescriptor = []
        meanEnergies = []
        if parameters.mode == "training"
            meanCrossAccumulators = []
        end
        meanAcceptanceRatio = []
        meanMaxDisplacement = []
        # Find the corresponding outputs
        for outputID in eachindex(outputs)
            # Save the outputID if the system names from input and output match
            if systemParms.systemName == outputs[outputID].systemParms.systemName
                append!(meanDescriptor, [outputs[outputID].descriptor])
                append!(meanEnergies, [outputs[outputID].energies])
                if parameters.mode == "training"
                    append!(meanCrossAccumulators, [outputs[outputID].crossAccumulators])
                end
                append!(meanAcceptanceRatio, [outputs[outputID].acceptanceRatio])
                append!(meanMaxDisplacement, [outputs[outputID].systemParms.delta])
            end
        end
        # Take averages from each worker
        meanDescriptor = mean(meanDescriptor)
        meanEnergies = mean(meanEnergies)
        if parameters.mode == "training"
            meanCrossAccumulators = mean(meanCrossAccumulators)
        end
        meanAcceptanceRatio = mean(meanAcceptanceRatio)
        meanMaxDisplacement = mean(meanMaxDisplacement)
        if parameters.mode == "training"
            systemOutput = MCAverages(meanDescriptor, meanEnergies, meanCrossAccumulators, 
            meanAcceptanceRatio, systemParms)
        else
            systemOutput = MCAverages(meanDescriptor, meanEnergies, nothing, 
            meanAcceptanceRatio, systemParms)
        end
        # Compute loss and print some output info
        println("       Acceptance ratio = ", round(meanAcceptanceRatio, digits=4))
        println("       Max displacement = ", round(meanMaxDisplacement, digits=4))
        if parameters.mode == "training" 
            meanLoss += loss(systemOutput.descriptor, refRDFs[systemId])
        end

        append!(systemOutputs, [systemOutput])
        # Write descriptors and energies
        name = systemParms.systemName
        if parameters.mode == "training" 
            writedescriptor("RDFNN-$(name)-iter-$(iterString).dat", systemOutput.descriptor, systemParms)
            writeenergies("energies-$(name)-iter-$(iterString).dat", systemOutput.energies, parameters, 10)
        else
            writedescriptor("RDFNN-$(name).dat", systemOutput.descriptor, systemParms)
            writeenergies("energies-$(name).dat", systemOutput.energies, parameters, 10)
        end
    end
    if parameters.mode == "training" 
        meanLoss /= length(systemParmsList)
        println("   \nAverage Loss = ", round(meanLoss, digits=8))
    end
    return(systemOutputs)
end

"""
function train!(parameters, systemParmsList, model, opt, refRDFs)

Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) training iterations
"""
function train!(parameters, systemParmsList, model, opt, refRDFs)
    # Run training iterations
    iteration = 1
    while iteration <= parameters.iters
        iterString = lpad(iteration, 2, '0')
        println("\nIteration $(iteration)...")
        
        # Prepare multi-reference inputs
        inputs = prepMCInputs(parameters, systemParmsList, model)
     
        # Run the simulation in parallel
        outputs = pmap(mcsample!, inputs)

        # Collect averages corresponding to each reference system
        systemOutputs = collectSystemAverages(outputs, refRDFs, systemParmsList, parameters, iterString)

        # Compute loss and the gradients
        lossGradients = []
        for (systemId, systemOutput) in enumerate(systemOutputs)
            systemParms = systemParmsList[systemId]    
            lossGradient = computeLossGradients(systemOutput.crossAccumulators, 
                                                systemOutput.descriptor, refRDFs[systemId], 
                                                model, systemParms)
            append!(lossGradients, [lossGradient])
            # Write descriptors and energies
            name = systemParms.systemName
            writedescriptor("RDFNN-$(name)-iter-$(iterString).dat", systemOutput.descriptor, systemParms)
            writeenergies("energies-$(name)-iter-$(iterString).dat", systemOutput.energies, parameters, 10)
        end
        # Average the gradients
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

"""
simulate!(parameters, systemParms, model)
Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) sampling
"""
function simulate!(parameters, systemParms, model)
    # Pack inputs
    input = MCSampleInput(parameters, systemParms, model)
    inputs = [input for worker in workers()]
    
    # Run the simulation in parallel
    outputs = pmap(mcsample!, inputs)

    # Collect averages corresponding to each reference system
    systemOutputs = collectSystemAverages(outputs, nothing, [systemParms], parameters, nothing)
    return
end