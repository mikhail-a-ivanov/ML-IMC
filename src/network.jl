using StaticArrays
using LinearAlgebra
using Flux
using BSON: @save, @load

"""
function energyGradients(symmFuncMatrix, model)

Computes all gradients of energy with respect
to all parameters in the given network
"""
function computeEnergyGradients(symmFuncMatrix, model)
    N = length(symmFuncMatrix[:, 1])
    energyGradients = []
    # Compute energy gradients
    gs = gradient(totalEnergy, symmFuncMatrix, model)
    # Loop over the gradients and collect them in the array
    nlayers = length(model)
    # Structure: gs[2][1][layerId][1 - weigths; 2 - biases]
    # Need to divide the gradients by the number of atoms
    # to get the average gradient per atomic subnet
    for layerGradients in gs[2][1] 
        weightGradients = layerGradients[1] ./ N
        append!(energyGradients, [weightGradients])
        biasGradients = layerGradients[2] ./ N
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
        gradients = -Float32(systemParms.β) .* (accumulator .- ensemble)
        append!(descriptorGradients, [gradients])
    end
    return(descriptorGradients)
end

function computeLossGradients(crossAccumulators, symmFuncMatrix, descriptorNN, descriptorref, model, systemParms)
    lossGradients = []
    ensembleCorrelations = computeEnsembleCorrelation(symmFuncMatrix, descriptorNN, model)
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
function buildNetwork!(NNParms)

Combines input arguments for neural network building
Note: updates NNParms.neurons
"""
function buildNetwork!(NNParms)
    # Add output layer to the NNParms.neurons
    push!(NNParms.neurons, 1)
    nlayers = length(NNParms.neurons)
    network = []
    for layerId in 2:nlayers
        append!(network, [(NNParms.neurons[layerId - 1], NNParms.neurons[layerId],
                getfield(Main, Symbol(NNParms.activation)))])
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
    for arg in args
        layer = Dense(arg...)
        append!(layers, [layer])
    end
    model = Chain(layers...)
    return(model)
end

"""
function modelInit(NNParms)

Generates a neural network with all the weights set to random value
from -1 to 1 and biases to zero
"""
function modelInit(NNParms)
    # Build initial model
    network = buildNetwork!(NNParms)
    println("Building a model...")
    model = buildchain(network...)
    println(model)
    println("   Number of layers: $(length(NNParms.neurons)) ")
    println("   Number of neurons in each layer: $(NNParms.neurons)")
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
    return (opt)
end

"""
train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)

Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) training iterations
"""
function train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    # Initialize the list of loss values
    losses = []
    # Run training iterations
    iteration = 1
    while iteration <= NNParms.iters
        iterString = lpad(iteration, 2, '0')
        println("Iteration $(iteration)...")
        # Create an empty array for input data
        # for each reference system
        nsystems = length(systemParmsList)
        inputSet = []
        for systemId in 1:nsystems
            append!(inputSet, [(model, globalParms, MCParms, NNParms, systemParmsList[systemId])])
        end 
        # Generate as many input sets as the number of workers
        # divided by the number of the reference systems
        nsets = Int(nworkers()/nsystems)
        inputs = []
        for setId in 1:nsets
            append!(inputs, inputSet)
        end
     
        # Run the simulation in parallel
        outputs = pmap(mcsample!, inputs)

        # Create empty arrays for output data
        NNRDFs = []
        energies = []
        meanAcceptanceRatios = []
        crossAccumulators = []
        G2MatrixAccumulators = []

        for systemId in 1:nsystems
            systemParms = systemParmsList[systemId]
            println("System $(systemParms.systemName):")
            NNRDF = mean([output[1] for output in outputs[systemId:length(inputSet):end]])
            append!(NNRDFs, [NNRDF])
            meanSystemEnergy = mean([output[2] for output in outputs[systemId:length(inputSet):end]])
            append!(energies, [meanSystemEnergy])
            meanAcceptanceRatio = mean([output[3] for output in outputs[systemId:length(inputSet):end]])
            println("Mean acceptance ratio = ", round(meanAcceptanceRatio, digits=4))
            append!(meanAcceptanceRatios, meanAcceptanceRatio)
            crossAccumulator = mean([output[4] for output in outputs[systemId:length(inputSet):end]])
            append!(crossAccumulators, crossAccumulator)
            G2MatrixAccumulator = mean([output[5] for output in outputs[systemId:length(inputSet):end]])
            append!(G2MatrixAccumulators, [G2MatrixAccumulator])
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
            writeRDF("RDFNN-$(name)-iter-$(iterString).dat", NNRDFs[systemId], systemParms)
            lossGradient = computeLossGradients(crossAccumulators[systemId], G2MatrixAccumulators[systemId],
                                                NNRDFs[systemId], refRDFs[systemId], model, systemParms)
            append!(lossGradients, [lossGradient])

            # Write averaged energies
            writeenergies("energies-$(name)-iter-$(iterString).dat", energies[systemId], MCParms, 1)
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

"""
simulate!(model, globalParms, MCParms, NNParms, systemParms)

Runs the Machine Learning enhanced Inverse Monte Carlo (ML-IMC) sampling
"""
function simulate!(model, globalParms, MCParms, NNParms, systemParms)
    # Pack inputs
    inputs = (model, globalParms, MCParms, NNParms, systemParms)
    
    # Run the simulation in parallel
    outputs = pmap(mcsample!, inputs)

    # Average the data from workers
    NNRDF = mean([output[1] for output in outputs])
    meanSystemEnergy = mean([output[2] for output in outputs])
    meanAcceptanceRatio = mean([output[3] for output in outputs])
    println("Mean acceptance ratio = ", round(meanAcceptanceRatio, digits=4))

    # Write averaged descriptor and energies
    name = systemParms.systemName
    writeRDF("RDFNN-$(name).dat", NNRDF, systemParms)
    writeenergies("energies-$(name).dat", meanSystemEnergy, MCParms, 1)
    return
end