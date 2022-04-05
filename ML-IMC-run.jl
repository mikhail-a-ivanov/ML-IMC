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

    # Read reference histogram
    bins, rdfref, histref = readRDF(parameters.rdfname)

    return(bins, rdfref, histref, parameters)
end

"""
function inputconfs(parameters)

Reads input configurations from XYZ file
"""
function inputconfs(parameters)
    xyz = readXYZ(parameters.xyzname)
    confs = xyz[2:end] # Omit the initial configuration
    return(confs)
end

function main()
    # Initialize inputs
    bins, rdfref, histref, parameters = inputdata()

    # Read XYZ file the first time
    confs = inputconfs(parameters)
    @assert parameters.N == length(confs[end]) "Given number of particles does not match with XYZ configuration!"

    # Make a copy to read at the start of each iterations
    refconfs = copy(confs)

    # Normalize the reference histogram to per particle histogram
    histref ./= parameters.N / 2 # Number of pairs divided by the number of particles

    # Initialize the model
    model = modelinit(histref, parameters)

    # Initialize RNG for random input frame selection
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

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
    println("Using $(parameters.paircorr) as a pair descriptor")
    println("Using $(parameters.activation) activation")
    println("Number of iterations: $(parameters.iters)")
    println("Optimizer type: $(parameters.optimizer)")
    println("Learning rate: $(parameters.η)")
    if parameters.optimizer == "Momentum"
        println("Momentum coefficient: $(parameters.μ)")
    end
    println("Starting at: ", startTime)
     
    # Run training iterations
    for i in 1:parameters.iters
        iterString = lpad(i, 2, '0')
        println("Iteration $i...")
        inputs = [(confs[rand(rng_xor, 1:length(confs))], parameters, model) 
                for worker in workers()]
     
        # Run the simulation in parallel
        outputs = pmap(mcrun!, inputs)

        pairdescriptorNN = mean([output[1] for output in outputs])
        energies = mean([output[2] for output in outputs])
        crossWeights = mean([output[3] for output in outputs])
        crossBiases = mean([output[4] for output in outputs])
        meanAcceptanceRatio = mean([output[5] for output in outputs])

        # Write averaged energies
        writeenergies("energies-iter-$(iterString).dat", energies, parameters, 10)

        # Write the model (before training!)
        @save "model-iter-$(iterString).bson" model

        println("Mean acceptance ratio = $(meanAcceptanceRatio)")
        if parameters.paircorr == "RDF"
            loss(pairdescriptorNN, rdfref)
            writedescriptor("rdfNN-iter-$(iterString).dat", pairdescriptorNN, bins)
            dLdw, dLdb = computeDerivatives(crossWeights, crossBiases, pairdescriptorNN, rdfref, model, parameters)
        elseif parameters.paircorr == "histogram"
            loss(pairdescriptorNN, histref)
            writedescriptor("histNN-iter-$(iterString).dat", pairdescriptorNN, bins)
            dLdw, dLdb = computeDerivatives(crossWeights, crossBiases, pairdescriptorNN, histref, model, parameters)
        end

        # Update the model
        updatemodel!(model, opt, dLdw, dLdb)

        # Load the reference configurations
        confs = copy(refconfs)
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