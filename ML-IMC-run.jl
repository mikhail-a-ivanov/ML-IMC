using Dates
using Statistics
using LinearAlgebra
using Distributed

BLAS.set_num_threads(1)

@everywhere begin
    include("src/distances.jl")
    include("src/readLJ.jl")
    include("src/ML-IMC.jl");
end

"""
function inputdata(xyzname, rdfname, inputname)

Reads input data
"""
function inputdata(xyzname, rdfname, inputname)
    # Load a configuration from XYZ file
    xyz = readXYZ(xyzname)
    conf = xyz[end]

    # Read reference histogram
    bins, rdfref, histref = readRDF(rdfname)

    # Read input parameters
    parameters = readinput(inputname)
    return(conf, bins, rdfref, histref, parameters)
end

function main(xyzname, rdfname, inputname, activation=identity, η=0.05, iters=5)
    conf, bins, rdfref, histref, parameters = inputdata(xyzname, rdfname, inputname)
    model = Dense(length(histref), 1, activation, bias=true)

    # Normalize the reference histogram to per particle histogram
    histref ./= length(conf)/2 # Number of pairs divided by the number of particles

    # Start the timer
    startTime = Dates.now()
    println("Running MC simulation on $(nworkers()) rank(s)...\n")
    println("Total number of steps: $(parameters.steps * nworkers() / 1E6)M")
    println("Number of iterations: $(iters)")
    println("Learning rate: $(η)")
    println("Starting at: ", startTime)
     
    # Prepare inputs
    for i in 1:iters
        iterString = lpad(i, 2, '0')
        println("Iteration $i...")
        input = conf, parameters, model
        inputs = [input for worker in workers()]
     
        # Run the simulation in parallel
        outputs = pmap(mcrun!, inputs)

        histNN = mean([output[1] for output in outputs])
        energies = mean([output[2] for output in outputs])
        crossWeights = mean([output[3] for output in outputs])
        crossBiases = mean([output[4] for output in outputs])
        meanAcceptanceRatio = mean([output[5] for output in outputs])

        println("Mean acceptance ratio = $(meanAcceptanceRatio)")
        loss(histNN, histref)

        # Write the histogram
        writehist("histNN-iter-$(iterString).dat", histNN, bins)

        # Write averaged energies
        writeenergies("energies-iter-$(iterString).dat", energies, parameters, 10)

        # Write the model (before training!)
        savemodel("model-iter-$(iterString).dat", model)

        # Training
        dLdw, dLdb = computeDerivatives(crossWeights, crossBiases, histNN, histref, model, parameters)
        updatemodel!(model, η, dLdw, dLdb)
    end

    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    println("Walltime: ", wallTime)
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main("mctraj-p001.xyz", "rdf-mean-p40.dat", "LJML-init.in", identity, 0.05, 10)
end