using Printf
using RandomNumbers
using Statistics
using StaticArrays
using LinearAlgebra
using Flux

include("distances.jl")

"""
struct inputParms

Fields:
box: box vector, Å
β: 1/(kB*T), reciprocal kJ/mol
Δ: max displacement, Å
steps: total number of steps
Eqsteps: equilibration steps
xyzout: XYZ output frequency
outfreq: output frequency
binWidth: histogram bin width, Å
Nbins: number of histogram bins
"""
struct inputParms
    box::SVector{3, Float64}
    β::Float64
    Δ::Float64  
    steps::Int
    Eqsteps::Int
    xyzout::Int 
    outfreq::Int
    binWidth::Float64
    Nbins::Int
end

"""
readinput(inputname)

Reads an input file for ML-IMC
and saves the data into the
inputParms struct
"""
function readinput(inputname)
    # Constants
    NA::Float64 = 6.02214076E23 # [mol-1]
    kB::Float64 = 1.38064852E-23 * NA / 1000 # [kJ/(mol*K)]
    # Has to define the variable outside of the main loop
    box = zeros(3)
    β::Float64 = 0.
    Δ::Float64 = 0.
    steps::Int = 0
    Eqsteps::Int = 0
    xyzout::Int = 0
    outfreq::Int = 0
    binWidth::Float64 = 0.
    Nbins::Int = 0
    file = open(inputname, "r")
    lines = readlines(file)
    for line in lines
        if length(line) > 0 && line[1] != '#'
            splittedLine = split(line)
            if splittedLine[1] == "box"
                box[1] = parse(Float64, splittedLine[3])
                box[2] = parse(Float64, splittedLine[4])
                box[3] = parse(Float64, splittedLine[5])
            elseif splittedLine[1] == "temperature"
                T = parse(Float64, splittedLine[3])
                β = 1/(kB * T)
            elseif splittedLine[1] == "delta"
                Δ = parse(Float64, splittedLine[3])
            elseif splittedLine[1] == "steps"
                steps = Int(parse(Float64, splittedLine[3]))
            elseif splittedLine[1] == "Eqsteps"
                Eqsteps = Int(parse(Float64, splittedLine[3]))
            elseif splittedLine[1] == "xyzout"
                xyzout = Int(parse(Float64, splittedLine[3]))
            elseif splittedLine[1] == "outfreq"
                outfreq = Int(parse(Float64, splittedLine[3]))
            elseif splittedLine[1] == "binWidth"
                binWidth = parse(Float64, splittedLine[3])
            elseif splittedLine[1] == "Nbins"
                Nbins = Int(parse(Float64, splittedLine[3]))
            end
        end
    end
    # Save parameters into the inputParms struct
    parameters = inputParms(box, β, Δ, steps, Eqsteps, xyzout, outfreq, binWidth, Nbins)
    return(parameters)
end

"""
function histpart!(distanceVector, hist, binWidth, recalculate=false)

Accumulates pair distances from a distance vector
(one particle) to a histogram
"""
function histpart!(distanceVector, hist, binWidth)
    N = length(distanceVector)
    @inbounds @fastmath for i in 1:N
        if distanceVector[i] != 0
            histIndex = floor(Int, 0.5 + distanceVector[i]/binWidth)
            if histIndex <= length(hist)
                hist[histIndex] += 1
            end
        end
    end
    return(hist)
end

"""
neuralenergy(hist, model)

Computes the potential energy of one particle
distance histogram using the neural network
"""
function neuralenergy(hist, model)
    E::Float64 = model(hist)[1]
    return(E)
end

"""
mcmove!(conf, E, step, parameters, model, histNN, rng)

Performs a Metropolis Monte Carlo
displacement move using a neural network
to predict energies from distance histograms
"""
function mcmove!(conf, distanceMatrix, crossWeights, crossBiases, E, step, parameters, model, histNN, rng)
    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(length(conf)))
    
    # Allocate the distance vector
    distanceVector = distanceMatrix[:, pointIndex]

    # Allocate and compute the histogram
    hist1 = zeros(parameters.Nbins)
    histpart!(distanceVector, hist1, parameters.binWidth)
    
    # Compute the energy
    E1 = neuralenergy(hist1, model)
    
    # Displace the particle
    dr = SVector{3, Float64}(parameters.Δ*(rand(rng, Float64) - 0.5), 
                             parameters.Δ*(rand(rng, Float64) - 0.5), 
                             parameters.Δ*(rand(rng, Float64) - 0.5))
    
    conf[pointIndex] += dr
    
    # Update distance and compute the new histogram
    hist2 = zeros(parameters.Nbins)
    updatedistance!(conf, parameters.box, distanceVector, pointIndex)
    histpart!(distanceVector, hist2, parameters.binWidth)
    
    # Compute the energy again
    E2 = neuralenergy(hist2, model)
    
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
        # Add the particle histogram to the total histogram
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                 histNN[i] += hist2[i] 
            end
            # Update cross correlation arrays
            crossCorrelation!(hist2, model, crossWeights, crossBiases)
        end
    else
        conf[pointIndex] -= dr
        # Add the particle histogram to the total histogram
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                 histNN[i] += hist1[i] 
            end
            # Update cross correlation arrays
            crossCorrelation!(hist1, model, crossWeights, crossBiases)
        end
    end
    return(conf, distanceMatrix, crossWeights, crossBiases, E, histNN, accepted)
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

    # Allocate the histogram
    histNN = zeros(parameters.Nbins)

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
        conf, distanceMatrix, crossWeights, crossBiases, E, histNN, accepted = 
            mcmove!(conf, distanceMatrix, crossWeights, crossBiases, E, step, parameters, model, histNN, rng_xor)
        acceptedTotal += accepted
        if step % parameters.outfreq == 0
            energies[Int(step/parameters.outfreq) + 1] = E
        end
    end
    acceptanceRatio = acceptedTotal / parameters.steps

    # Normalize the histogram
    histNN ./= Float32(prodDataPoints)

    # Normalize the cross correlation arrays 
    crossWeights ./= prodDataPoints
    crossBiases ./= prodDataPoints

    println("Acceptance ratio = $(acceptanceRatio)")
    return(histNN, energies, crossWeights, crossBiases, acceptanceRatio)
end

"""
crossCorrelation!(hist, model, crossWeights, crossBiases)

Updated cross correlation arrays
"""
function crossCorrelation!(hist, model, crossWeights, crossBiases)
    dHdw, dHdb = gradient(neuralenergy, hist, model)[2]
    crossWeights .+= (hist * dHdw) # Matrix Nbins x Nweights
    crossBiases .+= (hist .* dHdb) # Vector Nbins
    return(crossWeights, crossBiases)
end

"""
function computeDerivatives(crossWeights, crossBiases, histNN, model, parameters)

Compute dL/dl derivatives for training (based on IMC method)
"""
function computeDerivatives(crossWeights, crossBiases, histNN, histref, model, parameters)
    # Compute <dH/dl> * <S>
    dHdw, dHdb = gradient(neuralenergy, histNN, model)[2]
    productWeights = Float32.(histNN * dHdw)
    productBiases = Float32.(histNN .* dHdb)

    # Compute histogram gradients
    dSdw = -Float32(parameters.β) .* (crossWeights - productWeights)
    dSdb = -Float32(parameters.β) .* (crossBiases - productBiases)

    # Loss derivative
    dL = lossDerivative(histNN, histref)

    # Loss total gradient
    dLdw = zeros(length(model.weight))
    for i in 1:length(dLdw)
        dLdw[i] = 2 * dL[i] * sum(dSdw[i, :]) # take a column from dS/dw matrix
    end
    dLdb = sum(dL .* dSdb)
    return(dLdw, dLdb)
end

"""
function updatemodel!(model, rate, dLdw, dLdb)

Update model parameters using the calculated 
gradients multiplied by rate
"""
function updatemodel!(model, rate, dLdw, dLdb)
    Flux.Optimise.update!(model.weight, rate*dLdw')
    Flux.Optimise.update!(model.bias, rate*dLdb')
end

"""
function loss(histNN, histref)

Compute the error function
"""
function loss(histNN, histref)
    loss = zeros(length(histNN))
    for i in 1:length(loss)
        loss[i] = (histNN[i] - histref[i])^2
    end
    totalLoss = sum(loss)
    println("Loss = ", round(totalLoss, digits=8))
    return(totalLoss)
end

"""
function lossDerivative(histNN, histref)

Compute the loss derivative function for dL/dl calculation
"""
function lossDerivative(histNN, histref)
    loss = zeros(length(histNN))
    for i in 1:length(loss)
        loss[i] = 2*(histNN[i] - histref[i])
    end
    return(loss)
end

"""
function writehist(outname, hist, parameters)

Writes the histogram into a file
"""
function writehist(outname, hist, bins)
    io = open(outname, "w")
    print(io, "# r, Å; Histogram \n")
    print(io, @sprintf("%6.3f %12.3f", bins[1], 0), "\n")
    for i in 2:length(hist)
        print(io, @sprintf("%6.3f %12.3f", bins[i], hist[i]), "\n")
    end
    close(io)
end

"""
function savemodel(outname, model)

Saves model into a file
"""
function savemodel(outname, model)
    io = open(outname, "w")
    print(io, "# ML-IMC model: $(length(model.weight)) weights; $(length(model.bias)) biases\n")
    print(io, "# Weights\n")
    for weight in model.weight
        print(io, @sprintf("%12.8f", weight), "\n")
    end
    print(io, "# Biases\n")
    for bias in model.bias
        print(io, @sprintf("%12.8f", bias), "\n")
    end
end