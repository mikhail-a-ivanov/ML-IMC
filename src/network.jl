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
        opt = Momentum(NNParms.rate, NNParms.μ)
    elseif NNParms.optimizer == "Descent"
        opt = Descent(NNParms.rate)
    else
        opt = Descent(NNParms.rate)
        println("Other types of optimizers are currently not supported!")
    end
    return(opt)
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
    println("Running training in $(globalParms.descent) descent mode.")
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

        # Update the model or revert and update the learning rate
        if iteration == 1
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
        else
            # Update the model if the mean loss decreased or if
            # there are no restrictions on the loss decrease
            if losses[iteration] < losses[iteration - 1] || globalParms.descent == "unrestricted"
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
                #println("The loss has decreased, updating the model...")
                updatemodel!(model, opt, meanLossGradients)
                # Move on to the next iteration
                iteration += 1
            else
                println("The loss has increased!")
                # Reduce the rate and reinitialize the optimizer
                println("Multiplying the learning rate by $(NNParms.rateAdjust) and reinitializing the optimizer...")
                NNParms.rate *= NNParms.rateAdjust
                println("New learning rate: $(round(NNParms.rate, digits=16))")
                opt = optInit(NNParms)

                # Load the previous model and the gradients
                prevIterString = lpad((iteration - 1), 2, '0')
                println("Reverting to model-iter-$(prevIterString).bson...")
                @load "model-iter-$(prevIterString).bson" model 
                @load "gradients-iter-$(prevIterString).bson" meanLossGradients 

                println("Updating the model with the new optimizer...")
                updatemodel!(model, opt, meanLossGradients)

                # Remove the last loss value
                deleteat!(losses, iteration)

                println("Repeating iteration $(iteration)...")
            end
        end 
    end
    println("The training is finished!")
    return
end