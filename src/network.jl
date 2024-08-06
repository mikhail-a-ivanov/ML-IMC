using Flux
using Statistics
using BSON: @save, @load
using Dates

"""
struct MCSampleInput

Used for packaging mcsample! inputs
"""
struct MCSampleInput
    globalParms::GlobalParameters
    MCParms::MCparameters
    NNParms::NNparameters
    systemParms::SystemParameters
    model::Chain
end

"""
struct MCAverages

Used for packaging mcsample! outputs
"""
struct MCAverages
    descriptor::Vector{Float64}
    energies::Vector{Float64}
    crossAccumulators::Union{Nothing, Vector{Matrix{Float64}}}
    symmFuncMatrixAccumulator::Union{Nothing, Matrix{Float64}}
    acceptanceRatio::Float64
    systemParms::SystemParameters
    mutatedStepAdjust::Float64
end

"""
function energyGradients(symmFuncMatrix, model, NNParms)

Computes all gradients of energy with respect
to all parameters in the given network
"""
function computeEnergyGradients(symmFuncMatrix, model, NNParms)
    energyGradients = []
    # Compute energy gradients
    gs = gradient(totalEnergyScalar, symmFuncMatrix, model)
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
    return (crossCorrelations)
end

"""
function crossAccumulatorsInit(NNParms, systemParms)
Initialize cross correlation accumulator arrays
"""
function crossAccumulatorsInit(NNParms, systemParms)
    crossAccumulators = []
    nlayers = length(NNParms.neurons)
    for layerId in 2:nlayers
        if NNParms.bias
            append!(crossAccumulators,
                    [zeros(Float64, (systemParms.Nbins, NNParms.neurons[layerId - 1] * NNParms.neurons[layerId]))])
            append!(crossAccumulators, [zeros(Float64, (systemParms.Nbins, NNParms.neurons[layerId]))])
        else
            append!(crossAccumulators,
                    [zeros(Float64, (systemParms.Nbins, NNParms.neurons[layerId - 1] * NNParms.neurons[layerId]))])
        end
    end
    return (crossAccumulators)
end

"""
function updateCrossAccumulators(crossAccumulators, descriptor, model, NNParms)

Updates cross accumulators by performing element-wise summation
of the cross accumulators with the new cross correlation data
"""
function updateCrossAccumulators!(crossAccumulators, symmFuncMatrix, descriptor, model, NNParms)
    energyGradients = computeEnergyGradients(symmFuncMatrix, model, NNParms)
    newCrossCorrelations = computeCrossCorrelation(descriptor, energyGradients)
    for (cross, newCross) in zip(crossAccumulators, newCrossCorrelations)
        cross .+= newCross
    end
    return (crossAccumulators)
end

"""
function computeEnsembleCorrelation(descriptor, model, NNParms)

Computes correlations of the ensemble averages of the descriptor
and the energy gradients
"""
function computeEnsembleCorrelation(symmFuncMatrix, descriptor, model, NNParms)
    energyGradients = computeEnergyGradients(symmFuncMatrix, model, NNParms)
    ensembleCorrelations = computeCrossCorrelation(descriptor, energyGradients)
    return (ensembleCorrelations)
end

"""
function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)

Computes the gradients of the descriptor with respect to the network parameters
"""
function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)
    descriptorGradients = []
    for (accumulator, ensemble) in zip(crossAccumulators, ensembleCorrelations)
        gradients = -systemParms.β .* (accumulator - ensemble)
        append!(descriptorGradients, [gradients])
    end
    return (descriptorGradients)
end

"""
function computeLossGradients(crossAccumulators, symmFuncMatrix, descriptorNN, descriptorref, model, systemParms, NNParms)

Computes the final loss gradients
"""
function computeLossGradients(crossAccumulators, symmFuncMatrix, descriptorNN, descriptorref, model, systemParms,
                              NNParms)
    lossGradients = []
    ensembleCorrelations = computeEnsembleCorrelation(symmFuncMatrix, descriptorNN, model, NNParms)
    descriptorGradients = computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)
    # Compute derivative of loss with respect to the descriptor
    descriptorPoints = length(descriptorNN)
    dLdS = zeros(Float64, descriptorPoints)
    for i in 1:descriptorPoints
        dLdS[i] = 2 * (descriptorNN[i] - descriptorref[i])
    end
    for (gradient, parameters) in zip(descriptorGradients, Flux.params(model))
        lossGradient = dLdS' * gradient
        lossGradient = reshape(lossGradient, size(parameters))
        # Add the regularization contribution (2 * REGP * parameters)
        regLossGradient = @. parameters * 2 * NNParms.REGP
        lossGradient .+= regLossGradient
        append!(lossGradients, [lossGradient])
    end
    return (lossGradients)
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
function loss(descriptorNN, descriptorref, model, NNParms, meanMaxDisplacement)

Compute the error function
"""
function loss(descriptorNN, descriptorref, model, NNParms, meanMaxDisplacement)
    outname = "loss.out"
    io = open(outname, "a")
    strLoss = sum((descriptorNN - descriptorref) .^ 2)
    if NNParms.REGP > 0
        regLoss = 0.0
        for parameters in Flux.params(model)
            regLoss += NNParms.REGP * sum(abs2, parameters) # sum of squared abs values
        end
        println("Regularization Loss = ", round(regLoss; digits=8))
        println(io, "Regularization Loss = ", round(regLoss; digits=8))
        totalLoss = strLoss + regLoss
    else
        totalLoss = strLoss
        println("Regularization Loss = ", 0)
        println(io, "Regularization Loss = ", 0)
    end
    println("Descriptor Loss = ", round(strLoss; digits=8))
    println(io, "Descriptor Loss = ", round(strLoss; digits=8))
    println(io, "Total Loss = ", round(totalLoss; digits=8))
    # Abnormal max displacement is an indication
    # of a poor model, even if the total loss is low!
    # Low max displacement results in a severely
    # undersampled configuration - it becomes "stuck"
    # at the initial configuration
    println(io, "Max displacement = ", round(meanMaxDisplacement; digits=8))
    close(io)
    checkfile(outname)
    return (totalLoss)
end

"""
function buildNetwork!(NNParms)
Combines input arguments for neural network building
"""
function buildNetwork!(NNParms)
    nlayers = length(NNParms.neurons)
    network = []
    for layerId in 2:nlayers
        append!(network,
                [(NNParms.neurons[layerId - 1], NNParms.neurons[layerId],
                  getfield(Main, Symbol(NNParms.activations[layerId - 1])))])
    end
    return (network)
end

"""
function buildchain(args...)
Build a multilayered neural network
"""
function buildchain(NNParms, args...)
    layers = []
    for (layerId, arg) in enumerate(args)
        if NNParms.bias
            layer = Dense(arg...)
        else
            layer = Dense(arg...; bias=false)
        end
        append!(layers, [layer])
    end
    model = Chain(layers...)
    return (model)
end

"""
function modelInit(NNParms)
"""
function modelInit(NNParms)
    # Build initial model
    network = buildNetwork!(NNParms)
    println("Building a model...")
    model = buildchain(NNParms, network...)
    model = fmap(f64, model)
    println(model)
    #println(typeof(model))
    println("   Number of layers: $(length(NNParms.neurons)) ")
    println("   Number of neurons in each layer: $(NNParms.neurons)")
    parameterCount = 0
    for layer in model
        parameterCount += sum(length, Flux.params(layer))
    end
    println("   Total number of parameters: $(parameterCount)")
    println("   Using bias parameters: $(NNParms.bias)")
    return (model)
end

function modelInitOther(NNParms)
    # Build initial model
    network = buildNetworkOther!(NNParms)
    println("Building a model...")
    model = buildchain(NNParms, network...)
    model = fmap(f64, model)
    println(model)
    #println(typeof(model))
    println("   Number of layers: $(length(NNParms.neurons)) ")
    println("   Number of neurons in each layer: $(NNParms.neurons)")
    parameterCount = 0
    for layer in model
        parameterCount += sum(length, Flux.params(layer))
    end
    println("   Total number of parameters: $(parameterCount)")
    println("   Using bias parameters: $(NNParms.bias)")
    return (model)
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
    nsets = Int(nworkers() / nsystems)
    inputs = []
    for setId in 1:nsets
        append!(inputs, multiReferenceInput)
    end
    return (inputs)
end

"""
function collectSystemAverages(outputs, refRDFs, systemParmsList, globalParms, NNParms, model)
Collects averages from different workers corresponding to one reference system
"""
function collectSystemAverages(outputs, refRDFs, systemParmsList, globalParms, NNParms, model)
    meanLoss = 0.0
    systemOutputs = []

    systemLosses = []

    for (systemId, systemParms) in enumerate(systemParmsList)
        println("   System $(systemParms.systemName):")
        systemLoss = 0.0

        meanDescriptor = []
        meanEnergies = []
        if globalParms.mode == "training"
            meanCrossAccumulators = []
            meansymmFuncMatrixAccumulator = []
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
                    append!(meansymmFuncMatrixAccumulator, [outputs[outputID].symmFuncMatrixAccumulator])
                end
                append!(meanAcceptanceRatio, [outputs[outputID].acceptanceRatio])
                append!(meanMaxDisplacement, [outputs[outputID].mutatedStepAdjust])
            end
        end
        # Take averages from each worker
        meanDescriptor = mean(meanDescriptor)
        meanEnergies = mean(meanEnergies)
        if globalParms.mode == "training"
            meanCrossAccumulators = mean(meanCrossAccumulators)
            meansymmFuncMatrixAccumulator = mean(meansymmFuncMatrixAccumulator)
        end
        meanAcceptanceRatio = mean(meanAcceptanceRatio)
        meanMaxDisplacement = mean(meanMaxDisplacement)
        if globalParms.mode == "training"
            systemOutput = MCAverages(meanDescriptor, meanEnergies, meanCrossAccumulators,
                                      meansymmFuncMatrixAccumulator, meanAcceptanceRatio, systemParms,
                                      meanMaxDisplacement)
        else
            systemOutput = MCAverages(meanDescriptor, meanEnergies, nothing, nothing, meanAcceptanceRatio, systemParms,
                                      meanMaxDisplacement)
        end
        # Compute loss and print some output info
        println("       Acceptance ratio = ", round(meanAcceptanceRatio; digits=4))
        println("       Max displacement = ", round(meanMaxDisplacement; digits=4))
        if globalParms.mode == "training"
            systemLoss = loss(systemOutput.descriptor, refRDFs[systemId], model, NNParms, meanMaxDisplacement)
            meanLoss += systemLoss
            append!(systemLosses, systemLoss)
        end

        append!(systemOutputs, [systemOutput])
    end
    if globalParms.mode == "training"
        meanLoss /= length(systemParmsList)
        println("   \nTotal Average Loss = ", round(meanLoss; digits=8))
    end
    return (systemOutputs, systemLosses)
end

"""
function adaptiveGradientCoeffs(systemLosses)
Computes coefficients for gradient scaling based on loss information
"""
function adaptiveGradientCoeffs(systemLosses)
    gradientCoeffs = systemLosses ./ maximum(systemLosses)
    normCoeff = 1.0 / sum(gradientCoeffs)
    return (gradientCoeffs .* normCoeff)
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
            writeEnergies("energies-$(name)-iter-$(iterString).dat", systemOutput.energies, MCParms, systemParms, 1)
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

        @save "opt-iter-$(iterString).bson" opt
        checkfile("opt-iter-$(iterString).bson")

        @save "gradients-iter-$(iterString).bson" meanLossGradients
        checkfile("gradients-iter-$(iterString).bson")

        # Update the model
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
    systemOutputs, systemLosses = collectSystemAverages(outputs, nothing, [systemParms], globalParms, nothing, nothing)

    # Write descriptors and energies
    name = systemParms.systemName
    writeRDF("RDFNN-$(name).dat", systemOutputs[1].descriptor, systemParms)
    writeEnergies("energies-$(name).dat", systemOutputs[1].energies, MCParms, systemParms, 1)
    return
end
