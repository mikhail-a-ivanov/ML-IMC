using Flux
using Statistics
using BSON: @save, @load

"""
struct MCSampleInput
    
Used for packaging mcsample! inputs
"""
struct MCSampleInput
    globalParms
    MCParms
    NNParms
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
    G2MatrixAccumulator
    acceptanceRatio
    systemParms
end

"""
function energyGradients(symmFuncMatrix, model)

Computes all gradients of energy with respect
to all parameters in the given network
"""
function computeEnergyGradients(symmFuncMatrix, model)
    energyGradients = []
    # Compute energy gradients
    gs = gradient(totalEnergy, symmFuncMatrix, model)
    # Loop over the gradients and collect them in the array
    nlayers = length(model)
    # Structure: gs[2][1][layerId][1 - weigths; 2 - biases]
    for (layerId, layerGradients) in enumerate(gs[2][1]) 
        weightGradients = layerGradients[1] 
        append!(energyGradients, [weightGradients])
        biasGradients = layerGradients[2] 
        append!(energyGradients, [biasGradients])
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
function crossAccumulatorsInit(NNParms, systemParms)

Initialize cross correlation accumulator arrays
"""
function crossAccumulatorsInit(NNParms, systemParms)
    crossAccumulators = []
    nlayers = length(NNParms.neurons)
    for layerId in 2:nlayers
        append!(crossAccumulators, [zeros(Float32, (systemParms.Nbins, 
                NNParms.neurons[layerId - 1] * NNParms.neurons[layerId]))])
        append!(crossAccumulators, [zeros(Float32, (systemParms.Nbins, 
                NNParms.neurons[layerId]))])
    end
    return(crossAccumulators)
end

"""
function updateCrossAccumulators(crossAccumulators, descriptor, model)

Updates cross accumulators by performing element-wise summation
of the cross accumulators with the new cross correlation data
"""
function updateCrossAccumulators!(crossAccumulators, symmFuncMatrix, descriptor, model)
    energyGradients = computeEnergyGradients(symmFuncMatrix, model)
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
function computeEnsembleCorrelation(symmFuncMatrix, descriptor, model)
    energyGradients = computeEnergyGradients(symmFuncMatrix, model)
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
function computeLossGradients(crossAccumulators, symmFuncMatrix, descriptorNN, descriptorref, model, systemParms, NNParms)

Computes the final loss gradients
"""
function computeLossGradients(crossAccumulators, symmFuncMatrix, descriptorNN, descriptorref, model, systemParms, NNParms)
    lossGradients = []
    ensembleCorrelations = computeEnsembleCorrelation(symmFuncMatrix, descriptorNN, model)
    descriptorGradients = computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)
    # Compute derivative of loss with respect to the descriptor
    descriptorPoints = length(descriptorNN)
    dLdS = zeros(Float32, descriptorPoints)
    for i in 1:descriptorPoints
        dLdS[i] = 2*(descriptorNN[i] - descriptorref[i])
    end
    for (gradient, parameters) in zip(descriptorGradients, Flux.params(model))
        lossGradient = dLdS' * gradient
        lossGradient = reshape(lossGradient, size(parameters))
        # Add the regularization contribution (2 * REGP * parameters)
        if NNParms.REGP > 0
            regLossGradient = @. parameters * 2 * NNParms.REGP
            lossGradient += regLossGradient
        end
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
function loss(descriptorNN, descriptorref, model, NNParms)

Compute the error function
"""
function loss(descriptorNN, descriptorref, model, NNParms)
    strLoss = sum((descriptorNN - descriptorref).^2)
    regLoss = 0.
    for parameters in Flux.params(model)
        regLoss += NNParms.REGP * sum(abs2, parameters) # sum of squared abs values
    end
    println("Descriptor Loss = ", round(strLoss, digits=8))
    println("Regularization Loss = ", round(regLoss, digits=8))
    totalLoss = strLoss + regLoss
    return(totalLoss)
end

"""
function buildNetwork!(NNParms)
Combines input arguments for neural network building
"""
function buildNetwork!(NNParms)
    nlayers = length(NNParms.neurons)
    network = []
    for layerId in 2:nlayers
        append!(network, [(NNParms.neurons[layerId - 1], NNParms.neurons[layerId],
                getfield(Main, Symbol(NNParms.activations[layerId - 1])))])
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
        layer = Dense(arg...)
        append!(layers, [layer])
    end
    model = Chain(layers...)
    return(model)
end

"""
function modelInit(NNParms, globalParms)
Generates a neural network with zero weigths
in the first layer and random values in other layers
"""
function modelInit(NNParms, globalParms)
    # Build initial model
    network = buildNetwork!(NNParms)
    println("Building a model...")
    model = buildchain(network...)
    println(model)
    println("   Number of layers: $(length(NNParms.neurons)) ")
    println("   Number of neurons in each layer: $(NNParms.neurons)")
    nlayers = length(model.layers)
    # Initialize weights
    if globalParms.inputmodel == "zero"
        for (layerId, layer) in enumerate(model.layers)
            for column in eachrow(layer.weight)
                if layerId == 1
                    Flux.Optimise.update!(column, column)
                end
            end
        end
    end
    return(model)
end

"""
function optInit(NNParms)

Initializes the optimizer
"""
function optInit(NNParms)
    if NNParms.optimizer == "Momentum"
        opt = Momentum(NNParms.rate, NNParms.momentum)

    elseif NNParms.optimizer == "Descent"
        opt = Descent(NNParms.rate)

    elseif NNParms.optimizer == "Nesterov"
        opt = Nesterov(NNParms.rate, NNParms.momentum)

    elseif NNParms.optimizer == "RMSProp"
        opt = RMSProp(NNParms.rate, NNParms.momentum)

    elseif NNParms.optimizer == "Adam" 
        opt = Adam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "RAdam" 
        opt = RAdam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdaMax"
        opt = AdaMax(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdaGrad"
        opt = AdaGrad(NNParms.rate)

    elseif  NNParms.optimizer == "AdaDelta"
        opt = AdaDelta(NNParms.rate)
    
    elseif NNParms.optimizer == "AMSGrad"
        opt = AMSGrad(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "NAdam"
        opt = NAdam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdamW"
        opt = AdamW(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "OAdam"
        opt = OAdam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdaBelief"
        opt = AdaBelief(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    else
        opt = Descent(NNParms.rate)
        println(
            "Unsupported type of optimizer! \n
            Default optimizer is 'Descent' \n
            For more optimizers look at: https://fluxml.ai/Flux.jl/stable/training/optimisers/ \n")
    end
    return(opt)
end

"""
function prepMCInputs(globalParms, MCParms, NNParms, systemParmsList, model)
Prepares multi-reference inputs for mcsample! function
"""
function prepMCInputs(globalParms, MCParms, NNParms, systemParmsList, model)
    nsystems = length(systemParmsList)
    multiReferenceInput = []
    for systemId in 1:nsystems
        input = MCSampleInput(globalParms, MCParms, NNParms, systemParmsList[systemId], model)
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
function collectSystemAverages(outputs, refRDFs, systemParmsList, globalParms, NNParms, model)
Collects averages from different workers corresponding to one reference system
"""
function collectSystemAverages(outputs, refRDFs, systemParmsList, globalParms, NNParms, model)
    meanLoss = 0.
    systemOutputs = []
    for (systemId, systemParms) in enumerate(systemParmsList)
        println("   System $(systemParms.systemName):") 
        meanDescriptor = []
        meanEnergies = []
        if globalParms.mode == "training"
            meanCrossAccumulators = []
            meanG2MatrixAccumulator = []
        end
        meanAcceptanceRatio = []
        meanMaxDisplacement = []
        # Find the corresponding outputs
        for outputID in eachindex(outputs)
            # Save the outputID if the system names from input and output match
            if systemParms.systemName == outputs[outputID].systemParms.systemName
                append!(meanDescriptor, [outputs[outputID].descriptor])
                append!(meanEnergies, [outputs[outputID].energies])
                if globalParms.mode == "training"
                    append!(meanCrossAccumulators, [outputs[outputID].crossAccumulators])
                    append!(meanG2MatrixAccumulator, [outputs[outputID].G2MatrixAccumulator])
                end
                append!(meanAcceptanceRatio, [outputs[outputID].acceptanceRatio])
                append!(meanMaxDisplacement, [outputs[outputID].systemParms.Δ])
            end
        end
        # Take averages from each worker
        meanDescriptor = mean(meanDescriptor)
        meanEnergies = mean(meanEnergies)
        if globalParms.mode == "training"
            meanCrossAccumulators = mean(meanCrossAccumulators)
            meanG2MatrixAccumulator = mean(meanG2MatrixAccumulator)
        end
        meanAcceptanceRatio = mean(meanAcceptanceRatio)
        meanMaxDisplacement = mean(meanMaxDisplacement)
        if globalParms.mode == "training"
            systemOutput = MCAverages(meanDescriptor, meanEnergies, meanCrossAccumulators, 
            meanG2MatrixAccumulator, meanAcceptanceRatio, systemParms)
        else
            systemOutput = MCAverages(meanDescriptor, meanEnergies, nothing, 
            nothing, meanAcceptanceRatio, systemParms)
        end
        # Compute loss and print some output info
        println("       Acceptance ratio = ", round(meanAcceptanceRatio, digits=4))
        println("       Max displacement = ", round(meanMaxDisplacement, digits=4))
        if globalParms.mode == "training" 
            meanLoss += loss(systemOutput.descriptor, refRDFs[systemId], model, NNParms)
        end

        append!(systemOutputs, [systemOutput])
    end
    if globalParms.mode == "training" 
        meanLoss /= length(systemParmsList)
        println("   \nTotal Average Loss = ", round(meanLoss, digits=8))
    end
    return(systemOutputs)
end

"""
function train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) training iterations
"""
function train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    # Run training iterations
    iteration = 1
    while iteration <= NNParms.iters
        iterString = lpad(iteration, 2, '0')
        println("\nIteration $(iteration)...")
        
        # Prepare multi-reference inputs
        inputs = prepMCInputs(globalParms, MCParms, NNParms, systemParmsList, model)
     
        # Run the simulation in parallel
        outputs = pmap(mcsample!, inputs)

        # Collect averages corresponding to each reference system
        systemOutputs = collectSystemAverages(outputs, refRDFs, systemParmsList, globalParms, NNParms, model)

        # Compute loss and the gradients
        lossGradients = []
        for (systemId, systemOutput) in enumerate(systemOutputs)
            systemParms = systemParmsList[systemId]    
            lossGradient = computeLossGradients(systemOutput.crossAccumulators, 
                        systemOutput.G2MatrixAccumulator, systemOutput.descriptor, refRDFs[systemId], 
                        model, systemParms, NNParms)
            append!(lossGradients, [lossGradient])
            # Write descriptors and energies
            name = systemParms.systemName
            writeRDF("RDFNN-$(name)-iter-$(iterString).dat", systemOutput.descriptor, systemParms)
            writeenergies("energies-$(name)-iter-$(iterString).dat", systemOutput.energies, MCParms, 1)
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
function simulate!(model, globalParms, MCParms, NNParms, systemParms)

Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) sampling
"""
function simulate!(model, globalParms, MCParms, NNParms, systemParms)
    # Pack inputs
    input = MCSampleInput(globalParms, MCParms, NNParms, systemParms, model)
    inputs = [input for worker in workers()]
    
    # Run the simulation in parallel
    outputs = pmap(mcsample!, inputs)

    # Collect averages corresponding to each reference system
    systemOutputs = collectSystemAverages(outputs, nothing, [systemParms], globalParms, nothing, nothing)

    # Write descriptors and energies
    name = systemParms.systemName
    writeRDF("RDFNN-$(name).dat", systemOutputs[1].descriptor, systemParms)
    writeenergies("energies-$(name).dat", systemOutputs[1].energies, MCParms, 1)
    return
end