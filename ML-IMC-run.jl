using Dates
using Statistics
using LinearAlgebra
using Distributed
using BSON: @save, @load

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
function inputdata()
    # Read input parameters
    inputname = ARGS[1]
    parameters = readinput(inputname)

    # Load a configuration from XYZ file
    xyz = readXYZ(parameters.xyzname)
    println("Using the last recorded frame as an input configuration...")
    conf = xyz[end]

    # Read reference histogram
    bins, rdfref, histref = readRDF(parameters.rdfname)

    return(conf, bins, rdfref, histref, parameters)
end

function main()
    # Initialize inputs
    conf, bins, rdfref, histref, parameters = inputdata()

    # Normalize the reference histogram to per particle histogram
    histref ./= length(conf)/2 # Number of pairs divided by the number of particles

    # Initialize the model
    model = modelinit(histref, parameters)

    # Initialize the optimizer
    if parameters.optimizer == "Momentum"
        opt = Momentum(parameters.η, parameters.μ)
    elseif parameters.optimizer == "Descent"
        opt = Descent(parameters.η)
    else
        opt = Descent(parameters.η)
        println("Other types of optimizers are currently not supported!")
    end

    # Start the timer
    startTime = Dates.now()
    println("Running MC simulation on $(nworkers()) rank(s)...\n")
    println("Total number of steps: $(parameters.steps * nworkers() / 1E6)M")
    println("Number of equilibration steps per rank: $(parameters.Eqsteps / 1E6)M")
    println("Using $(parameters.activation) activation")
    println("Number of iterations: $(parameters.iters)")
    println("Optimizer type: $(parameters.optimizer)")
    println("Learning rate: $(parameters.η)")
    if parameters.optimizer == "Momentum"
        println("Momentum coefficient: $(parameters.μ)")
    end
    println("Starting at: ", startTime)
     
    # Prepare inputs
    for i in 1:parameters.iters
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
        @save "model-iter-$(iterString).bson" model

        # Training
        dLdw, dLdb = computeDerivatives(crossWeights, crossBiases, histNN, histref, model, parameters)
        updatemodel!(model, opt, dLdw, dLdb)
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
    main()
end