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
crossCorrelation!(hist, model, crossWeigths, crossBiases)

Updated cross correlation arrays
"""
function crossCorrelation!(hist, model, crossWeigths, crossBiases)
    dHdw, dHdb = gradient(neuralenergy, hist, model)[2]
    crossWeigths .+= (hist * dHdw) # Matrix Nbins x Nweights
    crossBiases .+= (hist .* dHdb) # Vector Nbins
    return(crossWeigths, crossBiases)
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
function mcmove!(conf, distanceMatrix, crossWeigths, crossBiases, E, step, parameters, model, histNN, rng)
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
            crossCorrelation!(hist2, model, crossWeigths, crossBiases)
        end
    else
        conf[pointIndex] -= dr
        # Add the particle histogram to the total histogram
        if step % parameters.outfreq == 0 && step > parameters.Eqsteps
            for i in 1:parameters.Nbins
                 histNN[i] += hist1[i] 
            end
            # Update cross correlation arrays
            crossCorrelation!(hist1, model, crossWeigths, crossBiases)
        end
    end
    return(conf, distanceMatrix, crossWeigths, crossBiases, E, histNN, accepted)
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
    dataPoints = Int(parameters.steps / parameters.outfreq)

    # Allocate and initialize the energy
    energies = zeros(dataPoints + 1)
    E = 0.

    # Allocate the histogram
    histNN = zeros(parameters.Nbins)

    # Initialize RNG
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Build the distance matrix
    distanceMatrix = builddistanceMatrix(conf, parameters.box)

    # Build the cross correlation arrays
    crossWeigths = zeros(Float32, parameters.Nbins, length(model.weight))
    crossBiases = zeros(Float32, parameters.Nbins)

    # Acceptance counter
    acceptedTotal = 0

    # Run MC simulation
    @inbounds @fastmath for step in 1:parameters.steps
        conf, distanceMatrix, crossWeigths, crossBiases, E, histNN, accepted = 
            mcmove!(conf, distanceMatrix, crossWeigths, crossBiases, E, step, parameters, model, histNN, rng_xor)
        acceptedTotal += accepted
        if step % parameters.outfreq == 0
            energies[Int(step/parameters.outfreq) + 1] = E
        end
    end
    acceptanceRatio = acceptedTotal / parameters.steps

    # Normalize the histogram
    histNN ./= Float32(dataPoints)

    # Normalize the cross correlation arrays
    crossWeigths ./= dataPoints
    crossBiases ./= dataPoints

    println("Acceptance ratio = $(acceptanceRatio)")
    return(histNN, energies, crossWeigths, crossBiases, acceptanceRatio)
end
