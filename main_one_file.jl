using Distributed

@everywhere module MlImc
using BSON: @load, @save
using Chemfiles
using Dates
using Distributed
using Flux
using Flux.NNlib: relu
using LinearAlgebra
using Printf
using RandomNumbers
using RandomNumbers.Xorshifts
using Statistics

BLAS.set_num_threads(4)

# -----------------------------------------------------------------------------
# --- Structs

struct G2
    eta::Float64
    rcutoff::Float64
    rshift::Float64
end

struct G3
    eta::Float64
    lambda::Float64
    zeta::Float64
    rcutoff::Float64
    rshift::Float64
end

struct G9
    eta::Float64
    lambda::Float64
    zeta::Float64
    rcutoff::Float64
    rshift::Float64
end

struct NeuralNetParameters
    G2Functions::Vector{G2}
    G3Functions::Vector{G3}
    G9Functions::Vector{G9}
    maxDistanceCutoff::Float64
    symmFunctionScaling::Float64
    neurons::Vector{Int}
    iters::Int
    bias::Bool
    activations::Vector{String}
    REGP::Float64
    optimizer::String
    rate::Float64
    momentum::Float64
    decay1::Float64
    decay2::Float64
end


struct SystemParameters
    systemName::String
    trajfile::String
    topname::String
    N::Int
    atomname::String
    rdfname::String
    Nbins::Int
    binWidth::Float64
    T::Float64
    β::Float64
    Δ::Float64
    targetAR::Float64
end


struct MonteCarloParameters
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    trajout::Int
    outfreq::Int
end


struct GlobalParameters
    systemFiles::Vector{String}
    symmetryFunctionFile::String
    mode::String
    outputMode::String
    modelFile::String
    gradientsFile::String
    optimizerFile::String
    adaptiveScaling::Bool
end


struct PreTrainingParameters
    PTsteps::Int64
    PToutfreq::Int64
    PTREGP::Float64
    PToptimizer::String
    PTrate::Float64
    PTmomentum::Float64
    PTdecay1::Float64
    PTdecay2::Float64
end


struct MonteCarloSampleInput
    globalParms::GlobalParameters
    MCParms::MonteCarloParameters
    NNParms::NeuralNetParameters
    systemParms::SystemParameters
    model::Chain
end

struct PreComputedInput
    NNParms::NeuralNetParameters
    systemParms::SystemParameters
    refRDF::Vector{Float64}
end


struct MonteCarloAverages
    descriptor::Vector{Float64}
    energies::Vector{Float64}
    crossAccumulators::Union{Nothing, Vector{Matrix{Float64}}}
    symmFuncMatrixAccumulator::Union{Nothing, Matrix{Float64}}
    acceptanceRatio::Float64
    systemParms::SystemParameters
    mutatedStepAdjust::Float64
end

struct ReferenceData
    distanceMatrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    PMF::Vector{Float64}
    G2Matrices::Vector{Matrix{Float64}}
    G3Matrices::Vector{Matrix{Float64}}
    G9Matrices::Vector{Matrix{Float64}}
end

# -----------------------------------------------------------------------------
# --- Helper Functions (Utils)

function print_symmetry_function_info(NNParms)
    if !isempty(NNParms.G2Functions)
        println("    G2 symmetry functions:")
        println("    eta, Å^-2; rcutoff, Å; rshift, Å")
        for G2Function in NNParms.G2Functions
            println("       ", G2Function)
        end
    end
    if !isempty(NNParms.G3Functions)
        println("    G3 symmetry functions:")
        println("    eta, Å^-2; lambda; zeta; rcutoff, Å; rshift, Å")
        for G3Function in NNParms.G3Functions
            println("       ", G3Function)
        end
    end
    if !isempty(NNParms.G9Functions)
        println("    G9 symmetry functions:")
        println("    eta, Å^-2; lambda; zeta; rcutoff, Å; rshift, Å")
        for G9Function in NNParms.G9Functions
            println("       ", G9Function)
        end
    end
    println("Maximum cutoff distance: $(NNParms.maxDistanceCutoff) Å")
    return println("Symmetry function scaling parameter: $(NNParms.symmFunctionScaling)")
end

function checkfile(filename)
    @assert isfile(filename) "Could not locate $(filename)!"
end

function readXTC(systemParms)
    checkfile(systemParms.trajfile)
    traj = Trajectory(systemParms.trajfile)
    return traj
end

function readPDB(systemParms)
    checkfile(systemParms.topname)
    pdb = Trajectory(systemParms.topname)
    return pdb
end

function writeRDF(outname, rdf, systemParms)
    bins = [bin * systemParms.binWidth for bin in 1:(systemParms.Nbins)]
    # Write the data
    io = open(outname, "w")
    print(io, "# System: $(systemParms.systemName)\n")
    print(io, "# RDF data ($(systemParms.atomname) - $(systemParms.atomname)) \n")
    print(io, "# r, Å; g(r); \n")
    for i in eachindex(rdf)
        print(io, @sprintf("%6.3f %12.3f", bins[i], rdf[i]), "\n")
    end
    close(io)
    return checkfile(outname)
end

function writeEnergies(outname, energies, MCParms, systemParms, slicing=1)
    steps = 0:(MCParms.outfreq * slicing):(MCParms.steps)
    io = open(outname, "w")
    print(io, "# System: $(systemParms.systemName)\n#")
    print(io, @sprintf("%8s %22s", " Step", "Total energy, kJ/mol"))
    print(io, "\n")
    for i in 1:length(energies[1:slicing:end])
        print(io, @sprintf("%9d %10.3f", steps[i], energies[1:slicing:end][i]), "\n")
    end
    close(io)
    return checkfile(outname)
end

function writeTraj(conf, box, systemParms, outname, mode='w')
    # Create an empty Frame object
    frame = Frame()
    # Set PBC vectors
    boxCenter = box ./ 2
    set_cell!(frame, UnitCell(box))
    # Add wrapped atomic coordinates to the frame
    for i in 1:(systemParms.N)
        wrappedAtomCoords = wrap!(UnitCell(frame), conf[:, i]) .+ boxCenter
        add_atom!(frame, Atom(systemParms.atomname), wrappedAtomCoords)
    end
    # Write to file
    Trajectory(outname, mode) do traj
        return write(traj, frame)
    end
    checkfile(outname)
    return
end

function readRDF(rdfname)
    checkfile(rdfname)
    file = open(rdfname, "r")
    lines = readlines(file)
    ncomments = 2
    nlines = length(lines) - ncomments
    bins = zeros(nlines)
    rdf = zeros(nlines)
    for i in (1 + ncomments):length(lines)
        rdfline = split(lines[i])
        if rdfline[1] != "#"
            bins[i - ncomments] = parse(Float64, rdfline[1])
            rdf[i - ncomments] = parse(Float64, rdfline[2])
        end
    end
    close(file)
    return (bins, rdf)
end

function buildDistanceMatrixChemfiles(frame)
    N = length(frame)
    distanceMatrix = Matrix{Float64}(undef, N, N)
    @inbounds for i in 1:N
        @inbounds for j in 1:N
            distanceMatrix[i, j] = distance(frame, i - 1, j - 1)
        end
    end
    return (distanceMatrix)
end

function updateDistance!(frame, distanceVector, pointIndex)
    @fastmath @inbounds for i in 0:(length(distanceVector) - 1)
        distanceVector[i + 1] = distance(frame, i, pointIndex - 1)
    end
    return (distanceVector)
end

function distanceCutoff(distance, rcutoff=6.0)
    if distance > rcutoff
        return (0.0)
    else
        return (0.5 * (cos(π * distance / rcutoff) + 1.0))
    end
end

function computeDistanceComponent(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int32, round(dx / xsize))
    return dx
end

function computeDirectionalVector(r1, r2, box)
    return map(computeDistanceComponent, r1, r2, box)
end

function computeSquaredDistanceComponent(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int32, round(dx / xsize))
    return dx^2
end

function computeDistance(r1, r2, box)
    return sqrt.(reduce(+, map(computeSquaredDistanceComponent, r1, r2, box)))
end

function computeDistanceVector(r1, coordinates, box)
    return vec(sqrt.(sum(broadcast(computeSquaredDistanceComponent, r1, coordinates, box); dims=1)))
end

function buildDistanceMatrix(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

    distanceMatrix = Matrix{Float64}(undef, N, N)
    for i in 1:N
        distanceMatrix[i, :] = computeDistanceVector(coordinates[:, i], coordinates, box)
    end

    return (distanceMatrix)
end

function reportOpt(opt)
    println("Optimizer type: $(typeof(opt))")
    println("   Parameters:")
    for name in fieldnames(typeof(opt))
        if String(name) != "state" && String(name) != "velocity"
            if String(name) == "eta"
                println("       Learning rate: $(getfield(opt, (name)))")
            elseif String(name) == "beta"
                println("       Decays: $(getfield(opt, (name)))")
            elseif String(name) == "velocity"
                println("       Velocity: $(getfield(opt, (name)))")
            elseif String(name) == "beta"
                println("       Beta: $(getfield(opt, (name)))")
            elseif String(name) == "rho"
                println("       Momentum coefficient: $(getfield(opt, (name)))")
            end
        end
    end
end

# -----------------------------------------------------------------------------
# --- Initialization

function parametersInit()
    # Read the input name from the command line argument
    if length(ARGS) > 0
        inputname = ARGS[1]
        # Check if no input file is provided
        # or if ML-IMC is started from jupyter
        # in the latter case the first argument has "json" extension
    end
    if length(ARGS) == 0 || occursin("json", ARGS[1])
        println("No input file was provided!")
        println("Trying to read input data from ML-IMC-init.in")
        inputname = "ML-IMC-init.in"
    end

    # Constants
    NA::Float64 = 6.02214076 # [mol-1] * 10^-23
    kB::Float64 = 1.38064852 * NA / 1000 # [kJ/(mol*K)]

    # Read the input file
    checkfile(inputname)
    file = open(inputname, "r")
    lines = readlines(file)
    splittedLines = [split(line) for line in lines]

    # Make a list of field names
    globalFields = [String(field) for field in fieldnames(GlobalParameters)]
    MCFields = [String(field) for field in fieldnames(MonteCarloParameters)]
    NNFields = [String(field) for field in fieldnames(NeuralNetParameters)]
    preTrainFields = [String(field) for field in fieldnames(PreTrainingParameters)]

    # Input variable arrays
    globalVars = []
    MCVars = []
    NNVars = []
    preTrainVars = []
    G2s = []
    G3s = []
    G9s = []

    # Loop over fieldnames and fieldtypes and over splitted lines
    # Global parameters
    for (field, fieldtype) in zip(globalFields, fieldtypes(GlobalParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if field == "systemFiles"
                    systemFiles = []
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(systemFiles, [strip(element, ',')])
                        elseif element == "#"
                            break
                        end
                    end
                    append!(globalVars, [systemFiles])
                else
                    if fieldtype != String
                        append!(globalVars, parse(fieldtype, line[3]))
                    else
                        append!(globalVars, [line[3]])
                    end
                end
            end
        end
    end
    globalParms = GlobalParameters(globalVars...)

    # Symmetry functions
    maxDistanceCutoff = 0.0
    scalingFactor = 1.0
    checkfile(globalParms.symmetryFunctionFile)
    symmetryFunctionFile = open(globalParms.symmetryFunctionFile, "r")
    symmetryFunctionLines = readlines(symmetryFunctionFile)
    splittedSymmetryFunctionLines = [split(line) for line in symmetryFunctionLines]
    for line in splittedSymmetryFunctionLines
        if length(line) != 0 && line[1] == "G2"
            G2Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G2))
                append!(G2Parameters, parse(fieldtype, line[fieldIndex + 1]))
            end
            append!(G2s, [G2(G2Parameters...)])
        end
        if length(line) != 0 && line[1] == "G3"
            G3Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G3))
                append!(G3Parameters, parse(fieldtype, line[fieldIndex + 1]))
            end
            append!(G3s, [G3(G3Parameters...)])
        end
        if length(line) != 0 && line[1] == "G9"
            G9Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G9))
                append!(G9Parameters, parse(fieldtype, line[fieldIndex + 1]))
            end
            append!(G9s, [G9(G9Parameters...)])
        end
        if length(line) != 0 && line[1] == "scaling"
            scalingFactor = parse(Float64, line[3])
        end
    end
    # Set maximum distance cutoff
    for G2Function in G2s
        if maxDistanceCutoff < G2Function.rcutoff
            maxDistanceCutoff = G2Function.rcutoff
        end
    end
    for G3Function in G3s
        if maxDistanceCutoff < G3Function.rcutoff
            maxDistanceCutoff = G3Function.rcutoff
        end
    end
    for G9Function in G9s
        if maxDistanceCutoff < G9Function.rcutoff
            maxDistanceCutoff = G9Function.rcutoff
        end
    end

    # MC parameters
    for (field, fieldtype) in zip(MCFields, fieldtypes(MonteCarloParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if fieldtype != String
                    append!(MCVars, parse(fieldtype, line[3]))
                else
                    append!(MCVars, [line[3]])
                end
            end
        end
    end
    MCParms = MonteCarloParameters(MCVars...)

    # Loop over fieldnames and fieldtypes and over splitted lines
    for (field, fieldtype) in zip(NNFields, fieldtypes(NeuralNetParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if field == "neurons"
                    inputNeurons = length(G2s) + length(G3s) + length(G9s)
                    neurons = [inputNeurons]
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(neurons, parse(Int, strip(element, ',')))
                        elseif element == "#"
                            break
                        end
                    end
                    append!(NNVars, [neurons])
                elseif field == "activations"
                    activations = []
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(activations, [strip(element, ',')])
                        elseif element == "#"
                            break
                        end
                    end
                    append!(NNVars, [activations])
                else
                    if fieldtype != String
                        append!(NNVars, parse(fieldtype, line[3]))
                    else
                        append!(NNVars, [line[3]])
                    end
                end
            end
        end
    end
    NNParms = NeuralNetParameters(G2s, G3s, G9s, maxDistanceCutoff, scalingFactor, NNVars...)

    # Pre-training parameters
    for (field, fieldtype) in zip(preTrainFields, fieldtypes(PreTrainingParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if fieldtype != String
                    append!(preTrainVars, parse(fieldtype, line[3]))
                else
                    append!(preTrainVars, [line[3]])
                end
            end
        end
    end
    preTrainParms = PreTrainingParameters(preTrainVars...)

    # Read system input files
    systemParmsList = [] # list of systemParameters structs
    systemFields = [String(field) for field in fieldnames(SystemParameters)]
    for inputname in globalParms.systemFiles
        systemVars = []
        checkfile(inputname)
        file = open(inputname, "r")
        lines = readlines(file)
        splittedLines = [split(line) for line in lines]
        for (field, fieldtype) in zip(systemFields, fieldtypes(SystemParameters))
            for line in splittedLines
                if length(line) != 0 && field == line[1]
                    if field == "T"
                        T = parse(fieldtype, line[3])
                        β = 1 / (kB * T)
                        append!(systemVars, T)
                        append!(systemVars, β)
                    elseif field == "topname"
                        topname = [line[3]]
                        checkfile(topname[1])
                        pdb = Trajectory("$(topname[1])")
                        pdbFrame = read(pdb)
                        N = length(pdbFrame)
                        atomname = name(Atom(pdbFrame, 1))
                        append!(systemVars, topname)
                        append!(systemVars, N)
                        append!(systemVars, [atomname])
                    elseif field == "rdfname"
                        rdfname = [line[3]]
                        bins, rdf = readRDF("$(rdfname[1])")
                        Nbins = length(bins)
                        binWidth = bins[1]
                        append!(systemVars, [rdfname[1]])
                        append!(systemVars, Nbins)
                        append!(systemVars, binWidth)
                    else
                        if fieldtype != String
                            append!(systemVars, parse(fieldtype, line[3]))
                        else
                            append!(systemVars, [line[3]])
                        end
                    end
                end
            end
        end
        systemParms = SystemParameters(systemVars...)
        append!(systemParmsList, [systemParms])
    end

    if globalParms.mode == "training"
        println("Running ML-IMC in the training mode.")
    else
        println("Running ML-IMC in the simulation mode.")
    end

    return (globalParms, MCParms, NNParms, preTrainParms, systemParmsList)
end

function inputInit(globalParms, NNParms, preTrainParms, systemParmsList)
    # Read reference data
    refRDFs = []
    for systemParms in systemParmsList
        bins, refRDF = readRDF(systemParms.rdfname)
        append!(refRDFs, [refRDF])
    end

    # Initialize the model and the optimizer
    if globalParms.modelFile == "none"
        # Initialize the model
        println("Initializing a new neural network with random weigths")

        model = modelInit(NNParms)

        if globalParms.optimizerFile != "none"
            println("Ignoring given optimizer filename...")
        end
        if globalParms.gradientsFile != "none"
            println("Ignoring given gradients filename...")
        end
        # Run pre-training if no initial model is given
        opt = optInit(preTrainParms)
        # Restart the training
    else
        # Loading the model
        checkfile(globalParms.modelFile)
        println("Reading model from $(globalParms.modelFile)")
        @load globalParms.modelFile model
        if globalParms.mode == "training"
            # Either initialize the optimizer or read from a file
            if globalParms.optimizerFile != "none"
                checkfile(globalParms.optimizerFile)
                println("Reading optimizer state from $(globalParms.optimizerFile)")
                @load globalParms.optimizerFile opt
            else
                opt = optInit(NNParms)
            end
            # Optionally read gradients from a file
            if globalParms.gradientsFile != "none"
                checkfile(globalParms.gradientsFile)
                println("Reading gradients from $(globalParms.gradientsFile)")
                @load globalParms.gradientsFile meanLossGradients
            end
            # Update the model if both opt and gradients are restored
            if globalParms.optimizerFile != "none" && globalParms.gradientsFile != "none"
                println("\nUsing the restored gradients and optimizer to update the current model...\n")
                updatemodel!(model, opt, meanLossGradients)

                # Skip updating if no gradients are provided
            elseif globalParms.optimizerFile != "none" && globalParms.gradientsFile == "none"
                println("\nNo gradients were provided, rerunning the training iteration with the current model and restored optimizer...\n")

                # Update the model if gradients are provided without the optimizer:
                # valid for optimizer that do not save their state, e.g. Descent,
                # otherwise might produce unexpected results
            elseif globalParms.optimizerFile == "none" && globalParms.gradientsFile != "none"
                println("\nUsing the restored gradients with reinitialized optimizer to update the current model...\n")
                updatemodel!(model, opt, meanLossGradients)
            else
                println("\nNeither gradients nor optimizer were provided, rerunning the training iteration with the current model...\n")
            end
        end
    end

    if globalParms.mode == "training"
        return (model, opt, refRDFs)
    else
        return (model)
    end
end

function optInit(NNParms::NeuralNetParameters)
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

    elseif NNParms.optimizer == "AdaDelta"
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
        println("Unsupported type of optimizer! \n
                Default optimizer is 'Descent' \n
                For more optimizers look at: https://fluxml.ai/Flux.jl/stable/training/optimisers/ \n")
    end
    return (opt)
end

function optInit(preTrainParms::PreTrainingParameters)
    if preTrainParms.PToptimizer == "Momentum"
        opt = Momentum(preTrainParms.PTrate, preTrainParms.PTmomentum)

    elseif preTrainParms.PToptimizer == "Descent"
        opt = Descent(preTrainParms.PTrate)

    elseif preTrainParms.PToptimizer == "Nesterov"
        opt = Nesterov(preTrainParms.PTrate, preTrainParms.PTmomentum)

    elseif preTrainParms.PToptimizer == "RMSProp"
        opt = RMSProp(preTrainParms.PTrate, preTrainParms.PTmomentum)

    elseif preTrainParms.PToptimizer == "Adam"
        opt = Adam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "RAdam"
        opt = RAdam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdaMax"
        opt = AdaMax(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdaGrad"
        opt = AdaGrad(preTrainParms.PTrate)

    elseif preTrainParms.PToptimizer == "AdaDelta"
        opt = AdaDelta(preTrainParms.PTrate)

    elseif preTrainParms.PToptimizer == "AMSGrad"
        opt = AMSGrad(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "NAdam"
        opt = NAdam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdamW"
        opt = AdamW(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "OAdam"
        opt = OAdam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdaBelief"
        opt = AdaBelief(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    else
        opt = Descent(preTrainParms.PTrate)
        println("Unsupported type of optimizer! \n
                Default optimizer is 'Descent' \n
                For more optimizers look at: https://fluxml.ai/Flux.jl/stable/training/optimisers/ \n")
    end
    return (opt)
end

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

# -----------------------------------------------------------------------------
# --- Energy and Gradients

function compute_atomic_energy(inputlayer::AbstractVector{T}, model::Flux.Chain) where {T <: AbstractFloat}
    E = only(model(inputlayer))
    return E
end

function compute_system_total_energy_scalar(symm_func_matrix::AbstractMatrix{T},
                                            model::Flux.Chain) where {T <: AbstractFloat}
    return sum(compute_atomic_energy(row, model) for row in eachrow(symm_func_matrix))
end

function update_system_energies_vector(symm_func_matrix::AbstractMatrix{T},
                                       model::Flux.Chain,
                                       indices_for_update::AbstractVector{Bool},
                                       previous_energies::AbstractVector{T}) where {T <: AbstractFloat}
    updated_energies = copy(previous_energies)
    update_indices = findall(indices_for_update)

    if !isempty(update_indices)
        new_energies = [compute_atomic_energy(symm_func_matrix[i, :], model) for i in update_indices]
        updated_energies[update_indices] .= new_energies
    end

    return updated_energies
end

function get_energies_update_mask(distance_vector::AbstractVector{T},
                                  nn_params)::Vector{Bool} where {T <: AbstractFloat}
    return distance_vector .< nn_params.maxDistanceCutoff
end

function init_system_energies_vector(symm_func_matrix::AbstractMatrix{T}, model::Flux.Chain) where {T <: AbstractFloat}
    return [compute_atomic_energy(row, model) for row in eachrow(symm_func_matrix)]
end

function computeEnergyGradients(symmFuncMatrix, model, NNParms)
    energyGradients = []
    # Compute energy gradients
    gs = gradient(compute_system_total_energy_scalar, symmFuncMatrix, model)
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
    return energyGradients
end

function computeCrossCorrelation(descriptor, energyGradients)
    crossCorrelations = []
    for gradient in energyGradients
        cross = descriptor * gradient[:]' # Matrix Nbins x Nparameters
        append!(crossCorrelations, [cross])
    end
    return (crossCorrelations)
end

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

function updateCrossAccumulators!(crossAccumulators, symmFuncMatrix, descriptor, model, NNParms)
    energyGradients = computeEnergyGradients(symmFuncMatrix, model, NNParms)
    newCrossCorrelations = computeCrossCorrelation(descriptor, energyGradients)
    for (cross, newCross) in zip(crossAccumulators, newCrossCorrelations)
        cross .+= newCross
    end
    return (crossAccumulators)
end

function computeEnsembleCorrelation(symmFuncMatrix, descriptor, model, NNParms)
    energyGradients = computeEnergyGradients(symmFuncMatrix, model, NNParms)
    ensembleCorrelations = computeCrossCorrelation(descriptor, energyGradients)
    return (ensembleCorrelations)
end

function computeDescriptorGradients(crossAccumulators, ensembleCorrelations, systemParms)
    descriptorGradients = []
    for (accumulator, ensemble) in zip(crossAccumulators, ensembleCorrelations)
        gradients = -systemParms.β .* (accumulator - ensemble)
        append!(descriptorGradients, [gradients])
    end
    return (descriptorGradients)
end

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

function updatemodel!(model, opt, lossGradients)
    for (gradient, parameters) in zip(lossGradients, Flux.params(model))
        Flux.Optimise.update!(opt, parameters, gradient)
    end
    return
end

# -----------------------------------------------------------------------------
# --- Training

function loss(descriptor_nn, descriptor_ref, model, nn_params, mean_max_displacement)
    squared_difference = sum((descriptor_nn .- descriptor_ref) .^ 2)

    # Calculate L2 regularization loss
    reg_loss = 0.0
    if nn_params.REGP > 0.0
        for parameters in Flux.params(model)
            reg_loss += nn_params.REGP * sum(abs2, parameters)
        end
    end

    total_loss = squared_difference + reg_loss

    println("  Regularization Loss = $(round(reg_loss; digits=8))")
    println("  Descriptor Loss = $(round(squared_difference; digits=8))")
    println("  Total Loss = $(round(total_loss; digits=8))")
    println("  Max displacement = $(round(mean_max_displacement; digits=8))")

    # Log loss to file
    outname = "training-loss-values.out"
    open(outname, "a") do io
        println(io, round(squared_difference; digits=8))
    end
    checkfile(outname)

    return total_loss
end

function buildNetwork!(NNParms)
    nlayers = length(NNParms.neurons)
    network = []
    for layerId in 2:nlayers
        append!(network,
                [(NNParms.neurons[layerId - 1], NNParms.neurons[layerId],
                  getproperty(Flux.NNlib, Symbol(NNParms.activations[layerId - 1])))]) # Изменение здесь
        # [(NNParms.neurons[layerId - 1], NNParms.neurons[layerId],
        #   getfield(Main, Symbol(NNParms.activations[layerId - 1])))])
    end
    return (network)
end

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

function prepMCInputs(globalParms, MCParms, NNParms, systemParmsList, model)
    nsystems = length(systemParmsList)
    multiReferenceInput = []
    for systemId in 1:nsystems
        input = MonteCarloSampleInput(globalParms, MCParms, NNParms, systemParmsList[systemId], model)
        append!(multiReferenceInput, [input])
    end
    nsets = Int(nworkers() / nsystems)
    inputs = []
    for setId in 1:nsets
        append!(inputs, multiReferenceInput)
    end
    return (inputs)
end

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
            systemOutput = MonteCarloAverages(meanDescriptor, meanEnergies, meanCrossAccumulators,
                                      meansymmFuncMatrixAccumulator, meanAcceptanceRatio, systemParms,
                                      meanMaxDisplacement)
        else
            systemOutput = MonteCarloAverages(meanDescriptor, meanEnergies, nothing, nothing, meanAcceptanceRatio, systemParms,
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

function adaptiveGradientCoeffs(systemLosses)
    gradientCoeffs = systemLosses ./ maximum(systemLosses)
    normCoeff = 1.0 / sum(gradientCoeffs)
    return (gradientCoeffs .* normCoeff)
end

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

# -----------------------------------------------------------------------------
# --- Simulation

function simulate!(model, globalParms, MCParms, NNParms, systemParms)
    # Pack inputs
    input = MonteCarloSampleInput(globalParms, MCParms, NNParms, systemParms, model)
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

# -----------------------------------------------------------------------------
# --- Symmetry Function

function combineSymmFuncMatrices(G2Matrix, G3Matrix, G9Matrix)
    if G3Matrix == [] && G9Matrix == []
        symmFuncMatrix = G2Matrix
    else
        # Combine all symmetry functions into a temporary array
        symmFuncMatrices = [G2Matrix, G3Matrix, G9Matrix]

        # Remove empty matrices
        filter!(x -> x != [], symmFuncMatrices)
        # Unpack symmetry functions and concatenate horizontally into a single matrix
        symmFuncMatrix = hcat(symmFuncMatrices...)
    end
    return (symmFuncMatrix)
end

function computeG2Element(distance, eta, rcutoff, rshift)::Float64
    if distance > 0.0
        return exp(-eta * (distance - rshift)^2) * distanceCutoff(distance, rcutoff)
    else
        return 0.0
    end
end

function computeG2(distances, eta, rcutoff, rshift)::Float64
    sum = 0
    @simd for distance in distances
        sum += computeG2Element(distance, eta, rcutoff, rshift)
    end
    return (sum)
end

function buildG2Matrix(distanceMatrix, NNParms)
    N = size(distanceMatrix)[1]
    G2Matrix = Matrix{Float64}(undef, N, length(NNParms.G2Functions))
    for i in 1:N
        distanceVector = distanceMatrix[i, :]
        for (j, G2Function) in enumerate(NNParms.G2Functions)
            eta = G2Function.eta
            rcutoff = G2Function.rcutoff
            rshift = G2Function.rshift
            G2Matrix[i, j] = computeG2(distanceVector, eta, rcutoff, rshift)
        end
    end
    if NNParms.symmFunctionScaling == 1.0
        return (G2Matrix)
    else
        return (G2Matrix .* NNParms.symmFunctionScaling)
    end
end

function updateG2Matrix!(G2Matrix, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)
    for i in 1:(systemParms.N)
        # Rebuild the whole G2 matrix column for the displaced particle
        if i == pointIndex
            for (j, G2Function) in enumerate(NNParms.G2Functions)
                eta = G2Function.eta
                rcutoff = G2Function.rcutoff
                rshift = G2Function.rshift
                G2Matrix[pointIndex, j] = computeG2(distanceVector2, eta, rcutoff, rshift) * NNParms.symmFunctionScaling
            end
            # Compute the change in G2 caused by the displacement of an atom
        else
            for (j, G2Function) in enumerate(NNParms.G2Functions)
                rcutoff = G2Function.rcutoff
                if 0.0 < distanceVector2[i] < rcutoff || 0.0 < distanceVector1[i] < rcutoff
                    eta = G2Function.eta
                    rshift = G2Function.rshift
                    G2_1 = computeG2Element(distanceVector1[i], eta, rcutoff, rshift)
                    G2_2 = computeG2Element(distanceVector2[i], eta, rcutoff, rshift)
                    ΔG2 = G2_2 - G2_1
                    G2Matrix[i, j] += ΔG2 * NNParms.symmFunctionScaling
                end
            end
        end
    end
    return (G2Matrix)
end

function computeCosAngle(coordinates, box, i, j, k, distance_ij, distance_ik)::Float64
    @assert i != j && i != k && k != j
    vector_0i = coordinates[:, i]
    vector_ij = computeDirectionalVector(vector_0i, coordinates[:, j], box)
    vector_ik = computeDirectionalVector(vector_0i, coordinates[:, k], box)
    cosAngle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)
    @assert -1.0 <= cosAngle <= 1.0
    return (cosAngle)
end

function computeTripletGeometry(coordinates, box, i, j, k, distance_ij, distance_ik)::Tuple{Float64, Float64}
    @assert i != j && i != k && k != j
    vector_0i = coordinates[:, i]
    vector_0j = coordinates[:, j]
    vector_0k = coordinates[:, k]
    distance_kj = computeDistance(vector_0k, vector_0j, box)
    vector_ij = computeDirectionalVector(vector_0i, vector_0j, box)
    vector_ik = computeDirectionalVector(vector_0i, vector_0k, box)
    cosAngle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)
    @assert -1.0 <= cosAngle <= 1.0
    return (cosAngle, distance_kj)
end

function computeG3element(cosAngle, distance_ij, distance_ik, distance_kj, rcutoff, eta, zeta, lambda, rshift)::Float64
    return ((1.0 + lambda * cosAngle)^zeta *
            exp(-eta * ((distance_ij - rshift)^2 + (distance_ik - rshift)^2 + (distance_kj - rshift)^2)) *
            distanceCutoff(distance_ij, rcutoff) *
            distanceCutoff(distance_ik, rcutoff) *
            distanceCutoff(distance_kj, rcutoff))
end

function computeG3(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)::Float64
    sum = 0.0
    @inbounds for k in eachindex(distanceVector)
        distance_ik = distanceVector[k]
        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distanceVector[j]
            if 0 < distance_ij < rcutoff && 0 < distance_ik < rcutoff
                cosAngle, distance_kj = computeTripletGeometry(coordinates, box, i, j, k, distance_ij, distance_ik)
                sum += computeG3element(cosAngle, distance_ij, distance_ik, distance_kj, rcutoff, eta, zeta, lambda,
                                        rshift)
            end
        end
    end
    return (2.0^(1.0 - zeta) * sum)
end

function buildG3Matrix(distanceMatrix, coordinates, box, NNParms)
    N = size(distanceMatrix)[1]
    G3Matrix = Matrix{Float64}(undef, N, length(NNParms.G3Functions))
    for i in 1:N
        distanceVector = distanceMatrix[i, :]
        for (j, G3Function) in enumerate(NNParms.G3Functions)
            eta = G3Function.eta
            lambda = G3Function.lambda
            zeta = G3Function.zeta
            rcutoff = G3Function.rcutoff
            rshift = G3Function.rshift
            G3Matrix[i, j] = computeG3(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)
        end
    end
    if NNParms.symmFunctionScaling == 1.0
        return (G3Matrix)
    else
        return (G3Matrix .* NNParms.symmFunctionScaling)
    end
end

function updateG3Matrix!(G3Matrix, coordinates1, coordinates2, box, distanceVector1, distanceVector2, systemParms,
                         NNParms, displacedAtomIndex)
    for selectedAtomIndex in 1:(systemParms.N)
        # Rebuild the whole G3 matrix column for the displaced atom
        if selectedAtomIndex == displacedAtomIndex
            for (G3Index, G3Function) in enumerate(NNParms.G3Functions)
                eta = G3Function.eta
                lambda = G3Function.lambda
                zeta = G3Function.zeta
                rcutoff = G3Function.rcutoff
                rshift = G3Function.rshift
                G3Matrix[selectedAtomIndex, G3Index] = computeG3(displacedAtomIndex, coordinates2, box, distanceVector2,
                                                                 rcutoff, eta, zeta, lambda, rshift) *
                                                       NNParms.symmFunctionScaling
            end
            # Compute the change in G3 caused by the displacement of an atom
            # New ijk triplet description
            # Central atom (i): selectedAtomIndex
            # Second atom (j): displacedAtomIndex
            # Third atom (k): thirdAtomIndex
        else
            for (G3Index, G3Function) in enumerate(NNParms.G3Functions)
                rcutoff = G3Function.rcutoff
                distance_ij_1 = distanceVector1[selectedAtomIndex]
                distance_ij_2 = distanceVector2[selectedAtomIndex]
                # Ignore if the selected atom is far from the displaced atom
                if 0.0 < distance_ij_2 < rcutoff || 0.0 < distance_ij_1 < rcutoff
                    eta = G3Function.eta
                    lambda = G3Function.lambda
                    zeta = G3Function.zeta
                    rshift = G3Function.rshift
                    # Accumulate differences for the selected atom
                    # over all third atoms
                    ΔG3 = 0.0
                    # Loop over all ik pairs
                    for thirdAtomIndex in 1:(systemParms.N)
                        # Make sure i != j != k
                        if thirdAtomIndex != displacedAtomIndex && thirdAtomIndex != selectedAtomIndex
                            # It does not make a difference whether
                            # coordinates2 or coordinates1 are used -
                            # both selectedAtom and thirdAtom have
                            # have the same coordinates in the old and
                            # the updated configuration
                            selectedAtom = coordinates2[:, selectedAtomIndex]
                            thirdAtom = coordinates2[:, thirdAtomIndex]
                            distance_ik = computeDistance(selectedAtom, thirdAtom, box)
                            # The current ik pair is fixed so if r_ik > rcutoff
                            # no change in this G3(i,j,k) is needed
                            if 0.0 < distance_ik < rcutoff
                                # Compute the contribution to the change
                                # from the old configuration
                                displacedAtom_1 = coordinates1[:, displacedAtomIndex]
                                displacedAtom_2 = coordinates2[:, displacedAtomIndex]
                                distance_kj_1 = computeDistance(displacedAtom_1, thirdAtom, box)
                                distance_kj_2 = computeDistance(displacedAtom_2, thirdAtom, box)
                                if 0.0 < distance_kj_1 < rcutoff || 0.0 < distance_kj_2 < rcutoff
                                    # Compute cos of angle
                                    vector_ij_1 = computeDirectionalVector(selectedAtom, displacedAtom_1, box)
                                    vector_ij_2 = computeDirectionalVector(selectedAtom, displacedAtom_2, box)
                                    vector_ik = computeDirectionalVector(selectedAtom, thirdAtom, box)
                                    cosAngle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                    cosAngle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)
                                    @assert -1.0 <= cosAngle1 <= 1.0
                                    @assert -1.0 <= cosAngle2 <= 1.0
                                    # Compute differences in G3
                                    G3_1 = computeG3element(cosAngle1, distance_ij_1, distance_ik, distance_kj_1,
                                                            rcutoff, eta, zeta, lambda, rshift)
                                    G3_2 = computeG3element(cosAngle2, distance_ij_2, distance_ik, distance_kj_2,
                                                            rcutoff, eta, zeta, lambda, rshift)
                                    ΔG3 += 2.0^(1.0 - zeta) * (G3_2 - G3_1)
                                end
                            end
                        end
                    end
                    # Apply the computed differences
                    G3Matrix[selectedAtomIndex, G3Index] += ΔG3 * NNParms.symmFunctionScaling
                end
            end
        end
    end
    return (G3Matrix)
end

function computeG9element(cosAngle, distance_ij, distance_ik, rcutoff, eta, zeta, lambda, rshift)::Float64
    return ((1.0 + lambda * cosAngle)^zeta *
            exp(-eta * ((distance_ij - rshift)^2 + (distance_ik - rshift)^2)) *
            distanceCutoff(distance_ij, rcutoff) *
            distanceCutoff(distance_ik, rcutoff))
end

function computeG9(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)::Float64
    sum = 0.0
    @inbounds for k in eachindex(distanceVector)
        distance_ik = distanceVector[k]
        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distanceVector[j]
            if 0 < distance_ij < rcutoff && 0 < distance_ik < rcutoff
                cosAngle = computeCosAngle(coordinates, box, i, j, k, distance_ij, distance_ik)
                sum += computeG9element(cosAngle, distance_ij, distance_ik, rcutoff, eta, zeta, lambda, rshift)
            end
        end
    end
    return (2.0^(1.0 - zeta) * sum)
end

function buildG9Matrix(distanceMatrix, coordinates, box, NNParms)
    N = size(distanceMatrix)[1]
    G9Matrix = Matrix{Float64}(undef, N, length(NNParms.G9Functions))
    for i in 1:N
        distanceVector = distanceMatrix[i, :]
        for (j, G9Function) in enumerate(NNParms.G9Functions)
            eta = G9Function.eta
            lambda = G9Function.lambda
            zeta = G9Function.zeta
            rcutoff = G9Function.rcutoff
            rshift = G9Function.rshift
            G9Matrix[i, j] = computeG9(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)
        end
    end
    if NNParms.symmFunctionScaling == 1.0
        return (G9Matrix)
    else
        return (G9Matrix .* NNParms.symmFunctionScaling)
    end
end

function updateG9Matrix!(G9Matrix, coordinates1, coordinates2, box, distanceVector1, distanceVector2, systemParms,
                         NNParms, displacedAtomIndex)
    for selectedAtomIndex in 1:(systemParms.N)
        # Rebuild the whole G9 matrix column for the displaced atom
        if selectedAtomIndex == displacedAtomIndex
            for (G9Index, G9Function) in enumerate(NNParms.G9Functions)
                eta = G9Function.eta
                lambda = G9Function.lambda
                zeta = G9Function.zeta
                rcutoff = G9Function.rcutoff
                rshift = G9Function.rshift
                G9Matrix[selectedAtomIndex, G9Index] = computeG9(displacedAtomIndex, coordinates2, box, distanceVector2,
                                                                 rcutoff, eta, zeta, lambda, rshift) *
                                                       NNParms.symmFunctionScaling
            end
            # Compute the change in G9 caused by the displacement of an atom
            # New ijk triplet description
            # Central atom (i): selectedAtomIndex
            # Second atom (j): displacedAtomIndex
            # Third atom (k): thirdAtomIndex
        else
            for (G9Index, G9Function) in enumerate(NNParms.G9Functions)
                rcutoff = G9Function.rcutoff
                distance_ij_1 = distanceVector1[selectedAtomIndex]
                distance_ij_2 = distanceVector2[selectedAtomIndex]
                # Ignore if the selected atom is far from the displaced atom
                if 0.0 < distance_ij_2 < rcutoff || 0.0 < distance_ij_1 < rcutoff
                    eta = G9Function.eta
                    lambda = G9Function.lambda
                    zeta = G9Function.zeta
                    rshift = G9Function.rshift
                    # Accumulate differences for the selected atom
                    # over all third atoms
                    ΔG9 = 0.0
                    # Loop over all ik pairs
                    for thirdAtomIndex in 1:(systemParms.N)
                        # Make sure i != j != k
                        if thirdAtomIndex != displacedAtomIndex && thirdAtomIndex != selectedAtomIndex
                            # It does not make a difference whether
                            # coordinates2 or coordinates1 are used -
                            # both selectedAtom and thirdAtom have
                            # have the same coordinates in the old and
                            # the updated configuration
                            selectedAtom = coordinates2[:, selectedAtomIndex]
                            thirdAtom = coordinates2[:, thirdAtomIndex]
                            distance_ik = computeDistance(selectedAtom, thirdAtom, box)
                            # The current ik pair is fixed so if r_ik > rcutoff
                            # no change in this G9(i,j,k) is needed
                            if 0.0 < distance_ik < rcutoff
                                # Compute the contribution to the change
                                # from the old configuration
                                displacedAtom_1 = coordinates1[:, displacedAtomIndex]
                                displacedAtom_2 = coordinates2[:, displacedAtomIndex]
                                # Compute cos of angle
                                vector_ij_1 = computeDirectionalVector(selectedAtom, displacedAtom_1, box)
                                vector_ij_2 = computeDirectionalVector(selectedAtom, displacedAtom_2, box)
                                vector_ik = computeDirectionalVector(selectedAtom, thirdAtom, box)
                                cosAngle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                cosAngle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)
                                @assert -1.0 <= cosAngle1 <= 1.0
                                @assert -1.0 <= cosAngle2 <= 1.0
                                # Compute differences in G9
                                G9_1 = computeG9element(cosAngle1, distance_ij_1, distance_ik, rcutoff, eta, zeta,
                                                        lambda, rshift)
                                G9_2 = computeG9element(cosAngle2, distance_ij_2, distance_ik, rcutoff, eta, zeta,
                                                        lambda, rshift)
                                ΔG9 += 2.0^(1.0 - zeta) * (G9_2 - G9_1)
                            end
                        end
                    end
                    # Apply the computed differences
                    G9Matrix[selectedAtomIndex, G9Index] += ΔG9 * NNParms.symmFunctionScaling
                end
            end
        end
    end
    return (G9Matrix)
end

# -----------------------------------------------------------------------------
# --- Monte Carlo

function hist!(distanceMatrix, hist, systemParms)
    for i in 1:(systemParms.N)
        @fastmath for j in 1:(i - 1)
            histIndex = floor(Int, 1 + distanceMatrix[i, j] / systemParms.binWidth)
            if histIndex <= systemParms.Nbins
                hist[histIndex] += 1
            end
        end
    end
    return (hist)
end

function normalizehist!(hist, systemParms, box)
    boxVolume = box[1] * box[2] * box[3]
    Npairs::Int = systemParms.N * (systemParms.N - 1) / 2
    bins = [bin * systemParms.binWidth for bin in 1:(systemParms.Nbins)]
    shellVolumes = [4 * π * systemParms.binWidth * bins[i]^2 for i in eachindex(bins)]
    rdfNorm = ones(Float64, systemParms.Nbins)
    for i in eachindex(rdfNorm)
        rdfNorm[i] = boxVolume / Npairs / shellVolumes[i]
    end
    hist .*= rdfNorm
    return (hist)
end

function wrapFrame!(frame, box, pointIndex)
    if (positions(frame)[1, pointIndex] < 0.0)
        positions(frame)[1, pointIndex] += box[1]
    end
    if (positions(frame)[1, pointIndex] > box[1])
        positions(frame)[1, pointIndex] -= box[1]
    end
    if (positions(frame)[2, pointIndex] < 0.0)
        positions(frame)[2, pointIndex] += box[2]
    end
    if (positions(frame)[2, pointIndex] > box[2])
        positions(frame)[2, pointIndex] -= box[2]
    end
    if (positions(frame)[3, pointIndex] < 0.0)
        positions(frame)[3, pointIndex] += box[3]
    end
    if (positions(frame)[3, pointIndex] > box[3])
        positions(frame)[3, pointIndex] -= box[3]
    end
end

function mcmove!(mcarrays, E, EpreviousVector, model, NNParms, systemParms, box, rng, mutatedStepAdjust)
    # Unpack mcarrays
    frame, distanceMatrix, G2Matrix1, G3Matrix1, G9Matrix1 = mcarrays
    # Optionally make a copy of the original coordinates
    if G3Matrix1 != [] || G9Matrix1 != []
        coordinates1 = copy(positions(frame))
    end

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(systemParms.N))

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Take a copy of the previous energy value
    E1 = copy(E)

    # Displace the particle
    dr = mutatedStepAdjust * (rand(rng, Float64, 3) .- 0.5)

    positions(frame)[:, pointIndex] .+= dr

    # Check if all coordinates inside simulation box with PBC
    wrapFrame!(frame, box, pointIndex)

    # Compute the updated distance vector
    point = positions(frame)[:, pointIndex]
    distanceVector2 = computeDistanceVector(point, positions(frame), box)

    # Acceptance counter
    accepted = 0

    # Get indexes of atoms for energy contribution update
    indexesForUpdate = get_energies_update_mask(distanceVector2, NNParms)

    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(G2Matrix1)
    updateG2Matrix!(G2Matrix2, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

    # Make a copy of the original angular matrices and update them
    G3Matrix2 = copy(G3Matrix1)
    if G3Matrix1 != []
        updateG3Matrix!(G3Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    G9Matrix2 = copy(G9Matrix1)
    if G9Matrix1 != []
        updateG9Matrix!(G9Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Combine symmetry function matrices accumulators
    symmFuncMatrix2 = combineSymmFuncMatrices(G2Matrix2, G3Matrix2, G9Matrix2)

    # Compute the energy again
    # E2 = totalEnergyScalar(G2Matrix2, model)
    newEnergyVector = update_system_energies_vector(symmFuncMatrix2, model, indexesForUpdate, EpreviousVector)
    E2 = sum(newEnergyVector)

    # Get energy difference
    ΔE = E2 - E1

    # Accept or reject the move
    if rand(rng, Float64) < exp(-ΔE * systemParms.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector2
        distanceMatrix[:, pointIndex] = distanceVector2
        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix2, G3Matrix2, G9Matrix2)
        return (mcarrays, E, newEnergyVector, accepted)
    else
        positions(frame)[:, pointIndex] .-= dr

        # Check if all coordinates inside simulation box with PBC
        wrapFrame!(frame, box, pointIndex)

        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix1, G3Matrix1, G9Matrix1)
        return (mcarrays, E, EpreviousVector, accepted)
    end
end

function mcsample!(input)
    # Unpack the inputs
    model = input.model
    globalParms = input.globalParms
    MCParms = input.MCParms
    NNParms = input.NNParms
    systemParms = input.systemParms

    mutatedStepAdjust = copy(systemParms.Δ)

    # Get the worker id and the output filenames
    if nprocs() == 1
        id = myid()
    else
        id = myid() - 1
    end
    idString = lpad(id, 3, '0')

    trajFile = "mctraj-p$(idString).xtc"
    pdbFile = "confin-p$(idString).pdb"

    # Initialize RNG
    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    if globalParms.mode == "training"
        # Take a random frame from the equilibrated trajectory
        traj = readXTC(systemParms)
        nframes = Int(size(traj)) - 1
        frameId = rand(rngXor, 1:nframes) # Don't take the first frame
        frame = deepcopy(read_step(traj, frameId))
    else
        # Read PDB data from the system.in file
        pdb = readPDB(systemParms)
        frame = deepcopy(read_step(pdb, 0))
    end

    # Get current box vectors
    box = lengths(UnitCell(frame))

    # Start writing MC trajectory
    if globalParms.outputMode == "verbose"
        writeTraj(positions(frame), box, systemParms, trajFile, 'w')
        writeTraj(positions(frame), box, systemParms, pdbFile, 'w')
    end

    # Get the number of data points
    totalDataPoints = Int(MCParms.steps / MCParms.outfreq)
    prodDataPoints = Int((MCParms.steps - MCParms.Eqsteps) / MCParms.outfreq)

    # Build the distance matrix
    distanceMatrix = buildDistanceMatrix(frame)

    # Build the symmetry function matrices
    G2Matrix = buildG2Matrix(distanceMatrix, NNParms)

    G3Matrix = []
    if length(NNParms.G3Functions) > 0
        G3Matrix = buildG3Matrix(distanceMatrix, positions(frame), box, NNParms)
    end

    G9Matrix = []
    if length(NNParms.G9Functions) > 0
        G9Matrix = buildG9Matrix(distanceMatrix, positions(frame), box, NNParms)
    end

    # Prepare a tuple of arrays that change duing the mcmove!
    mcarrays = (frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix)

    # Initialize the distance histogram accumulator
    histAccumulator = zeros(Float64, systemParms.Nbins)

    # Build the cross correlation arrays for training,
    # an additional distance histogram array
    # and the symmetry function matrix accumulator
    if globalParms.mode == "training"
        hist = zeros(Float64, systemParms.Nbins)
        G2MatrixAccumulator = zeros(size(G2Matrix))
        G3MatrixAccumulator = zeros(size(G3Matrix))
        G9MatrixAccumulator = zeros(size(G9Matrix))
        crossAccumulators = crossAccumulatorsInit(NNParms, systemParms)
    end

    # Combine symmetry function matrices
    symmFuncMatrix = combineSymmFuncMatrices(G2Matrix, G3Matrix, G9Matrix)

    # Initialize the starting energy and the energy array
    EpreviousVector = init_system_energies_vector(symmFuncMatrix, model)
    E = sum(EpreviousVector)
    energies = zeros(totalDataPoints + 1)
    energies[1] = E

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @fastmath for step in 1:(MCParms.steps)
        mcarrays, E, EpreviousVector, accepted = mcmove!(mcarrays, E, EpreviousVector, model, NNParms, systemParms, box,
                                                         rngXor, mutatedStepAdjust)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        # Perform MC step adjustment during the equilibration
        if MCParms.stepAdjustFreq > 0 && step % MCParms.stepAdjustFreq == 0 && step < MCParms.Eqsteps
            mutatedStepAdjust = stepAdjustment!(mutatedStepAdjust, systemParms, box, MCParms, acceptedIntermediate)
            acceptedIntermediate = 0
        end

        # Collect the output energies
        if step % MCParms.outfreq == 0
            energies[Int(step / MCParms.outfreq) + 1] = E
        end

        # MC trajectory output
        if globalParms.outputMode == "verbose"
            if step % MCParms.trajout == 0
                writeTraj(positions(mcarrays[1]), box, systemParms, trajFile, 'a')
            end
        end

        # Accumulate the distance histogram
        if step % MCParms.outfreq == 0 && step > MCParms.Eqsteps
            frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix = mcarrays
            # Update the cross correlation array during the training
            if globalParms.mode == "training"
                hist = hist!(distanceMatrix, hist, systemParms)
                histAccumulator .+= hist
                G2MatrixAccumulator .+= G2Matrix
                if G3Matrix != []
                    G3MatrixAccumulator .+= G3Matrix
                end
                if G9Matrix != []
                    G9MatrixAccumulator .+= G9Matrix
                end
                # Normalize the histogram to RDF
                normalizehist!(hist, systemParms, box)

                # Combine symmetry function matrices
                symmFuncMatrix = combineSymmFuncMatrices(G2Matrix, G3Matrix, G9Matrix)

                updateCrossAccumulators!(crossAccumulators, symmFuncMatrix, hist, model, NNParms)
                # Nullify the hist array for the next training iteration
                hist = zeros(Float64, systemParms.Nbins)
            else
                histAccumulator = hist!(distanceMatrix, histAccumulator, systemParms)
            end
        end
    end
    # Compute and report the final acceptance ratio
    acceptanceRatio = acceptedTotal / MCParms.steps

    # Unpack mcarrays and optionally normalize cross and G2Matrix accumulators
    frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix = mcarrays # might remove this line
    if globalParms.mode == "training"
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
        G2MatrixAccumulator ./= prodDataPoints
        if G3Matrix != []
            G3MatrixAccumulator ./= prodDataPoints
        end
        if G9Matrix != []
            G9MatrixAccumulator ./= prodDataPoints
        end
        symmFuncMatrixAccumulator = combineSymmFuncMatrices(G2MatrixAccumulator, G3MatrixAccumulator,
                                                            G9MatrixAccumulator)
    end

    # Normalize the sampled distance histogram
    histAccumulator ./= prodDataPoints
    normalizehist!(histAccumulator, systemParms, box)

    # Combine symmetry function matrices accumulators
    if globalParms.mode == "training"
        MCoutput = MonteCarloAverages(histAccumulator, energies, crossAccumulators, symmFuncMatrixAccumulator, acceptanceRatio,
                              systemParms, mutatedStepAdjust)
        return (MCoutput)
    else
        MCoutput = MonteCarloAverages(histAccumulator, energies, nothing, nothing, acceptanceRatio, systemParms,
                              mutatedStepAdjust)
        return (MCoutput)
    end
end

function stepAdjustment!(mutatedStepAdjust, systemParms, box, MCParms, acceptedIntermediate)
    acceptanceRatio = acceptedIntermediate / MCParms.stepAdjustFreq
    mutatedStepAdjust = acceptanceRatio * mutatedStepAdjust / systemParms.targetAR

    if mutatedStepAdjust > box[1]
        mutatedStepAdjust /= 2
    end

    return mutatedStepAdjust
end

# -----------------------------------------------------------------------------
# --- Pre Training

function computePMF(refRDF, systemParms)
    PMF = Array{Float64}(undef, systemParms.Nbins)
    repulsionRegion = refRDF .== 0
    repulsionPoints = length(repulsionRegion[repulsionRegion .!= 0])
    maxPMFIndex = repulsionPoints + 1
    maxPMF = -log(refRDF[maxPMFIndex]) / systemParms.β
    secondMaxPMF = -log(refRDF[maxPMFIndex + 1]) / systemParms.β
    diffPMF = maxPMF - secondMaxPMF

    for i in eachindex(PMF)
        if repulsionRegion[i]
            PMF[i] = maxPMF + diffPMF * (maxPMFIndex - i)
        else
            PMF[i] = -log(refRDF[i]) / systemParms.β
        end
    end
    return (PMF)
end

function updatehist!(hist, distanceVector1, distanceVector2, systemParms)
    @fastmath for i in 1:(systemParms.N)
        # Remove histogram entries corresponding to distanceVector1
        histIndex = floor(Int, 1 + distanceVector1[i] / systemParms.binWidth)
        if histIndex <= systemParms.Nbins
            hist[histIndex] -= 1
        end
        # Add histogram entries corresponding to distanceVector2
        histIndex = floor(Int, 1 + distanceVector2[i] / systemParms.binWidth)
        if histIndex <= systemParms.Nbins
            hist[histIndex] += 1
        end
    end
    return (hist)
end

function preComputeRefData(refDataInput::PreComputedInput)::ReferenceData
    distanceMatrices = []
    histograms = []
    G2Matrices = []
    G3Matrices = []
    G9Matrices = []

    # Unpack the input struct
    NNParms = refDataInput.NNParms
    systemParms = refDataInput.systemParms
    refRDF = refDataInput.refRDF

    println("Pre-computing distances and symmetry function matrices for $(systemParms.systemName)...")
    PMF = computePMF(refRDF, systemParms)

    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame

    for frameId in 1:nframes
        #println("Frame $(frameId)...")
        frame = read_step(traj, frameId)
        box = lengths(UnitCell(frame))
        coordinates = positions(frame)

        distanceMatrix = buildDistanceMatrix(frame)
        append!(distanceMatrices, [distanceMatrix])

        hist = zeros(Float64, systemParms.Nbins)
        hist = hist!(distanceMatrix, hist, systemParms)
        append!(histograms, [hist])

        G2Matrix = buildG2Matrix(distanceMatrix, NNParms)
        append!(G2Matrices, [G2Matrix])

        if length(NNParms.G3Functions) > 0
            G3Matrix = buildG3Matrix(distanceMatrix, coordinates, box, NNParms)
            append!(G3Matrices, [G3Matrix])
        end

        if length(NNParms.G9Functions) > 0
            G9Matrix = buildG9Matrix(distanceMatrix, coordinates, box, NNParms)
            append!(G9Matrices, [G9Matrix])
        end
    end
    # Save the output in the referenceData struct
    refData = ReferenceData(distanceMatrices, histograms, PMF, G2Matrices, G3Matrices, G9Matrices)
    return (refData)
end

function computePreTrainingLossGradients(ΔENN, ΔEPMF, symmFuncMatrix1, symmFuncMatrix2, model,
                                         preTrainParms::PreTrainingParameters, NNParms::NeuralNetParameters, verbose=false)
    parameters = Flux.params(model)
    loss = (ΔENN - ΔEPMF)^2

    outname = "pretraining-loss-values.out"
    open(outname, "a") do io
        println(io, round(loss; digits=8))
    end

    regloss = sum(parameters[1] .^ 2) * preTrainParms.PTREGP
    if verbose
        println("  Energy loss: $(round(loss, digits=8))")
        println("  PMF energy difference: $(round(ΔEPMF, digits=8))")
        println("  NN energy difference: $(round(ΔENN, digits=8))")
        println("  Regularization loss: $(round(regloss, digits=8))")
    end

    # Compute dL/dw
    ENN1Gradients = computeEnergyGradients(symmFuncMatrix1, model, NNParms)
    ENN2Gradients = computeEnergyGradients(symmFuncMatrix2, model, NNParms)
    gradientScaling = 2 * (ΔENN - ΔEPMF)

    lossGradient = @. gradientScaling * (ENN2Gradients - ENN1Gradients)
    regLossGradient = @. parameters * 2 * preTrainParms.PTREGP
    lossGradient += regLossGradient
    return (lossGradient)
end

function pretraining_move!(refData::ReferenceData, model, NNParms, systemParms, rng)
    # Pick a frame
    traj = readXTC(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame
    frameId::Int = rand(rng, 1:nframes)
    frame = read_step(traj, frameId)
    box = lengths(UnitCell(frame))
    # Pick a particle
    pointIndex::Int = rand(rng, 1:(systemParms.N))

    # Read reference data
    distanceMatrix = refData.distanceMatrices[frameId]
    hist = refData.histograms[frameId]
    PMF = refData.PMF

    # If no angular symmetry functions are provided, use G2 only
    if refData.G3Matrices == [] && refData.G9Matrices == []
        symmFuncMatrix1 = refData.G2Matrices[frameId]
    else
        # Make a copy of the original coordinates
        coordinates1 = copy(positions(frame))
        # Combine symmetry function matrices
        if refData.G3Matrices == []
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G9Matrices[frameId]]
        elseif refData.G9Matrices == []
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G3Matrices[frameId]]
        else
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G3Matrices[frameId], refData.G9Matrices[frameId]]
        end
        # Unpack symmetry functions and concatenate horizontally into a single matrix
        symmFuncMatrix1 = hcat(symmFuncMatrices...)
    end

    # Compute energy of the initial configuration
    ENN1Vector = init_system_energies_vector(symmFuncMatrix1, model)
    ENN1 = sum(ENN1Vector)
    EPMF1 = sum(hist .* PMF)

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Displace the particle
    dr = systemParms.Δ * (rand(rng, Float64, 3) .- 0.5)

    positions(frame)[:, pointIndex] .+= dr

    # Compute the updated distance vector
    point = positions(frame)[:, pointIndex]
    distanceVector2 = computeDistanceVector(point, positions(frame), box)

    # Update the histogram
    hist = updatehist!(hist, distanceVector1, distanceVector2, systemParms)

    # Get indexes for updating ENN
    indexesForUpdate = get_energies_update_mask(distanceVector2, NNParms)

    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(refData.G2Matrices[frameId])
    updateG2Matrix!(G2Matrix2, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

    # Make a copy of the original angular matrices and update them
    G3Matrix2 = []
    if refData.G3Matrices != []
        G3Matrix2 = copy(refData.G3Matrices[frameId])
        updateG3Matrix!(G3Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Make a copy of the original angular matrices and update them
    G9Matrix2 = []
    if refData.G9Matrices != []
        G9Matrix2 = copy(refData.G9Matrices[frameId])
        updateG9Matrix!(G9Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Combine symmetry function matrices accumulators
    symmFuncMatrix2 = combineSymmFuncMatrices(G2Matrix2, G3Matrix2, G9Matrix2)

    # Compute the NN energy again
    ENN2Vector = update_system_energies_vector(symmFuncMatrix2, model, indexesForUpdate, ENN1Vector)
    ENN2 = sum(ENN2Vector)
    EPMF2 = sum(hist .* PMF)

    # Get the energy differences
    ΔENN = ENN2 - ENN1
    ΔEPMF = EPMF2 - EPMF1

    # Revert the changes in the frame arrays
    positions(frame)[:, pointIndex] .-= dr
    hist = updatehist!(hist, distanceVector2, distanceVector1, systemParms)

    return (symmFuncMatrix1, symmFuncMatrix2, ΔENN, ΔEPMF)
end

function pretrain!(preTrainParms::PreTrainingParameters, NNParms, systemParmsList, model, opt, refRDFs)
    println("\nRunning $(preTrainParms.PTsteps) steps of pre-training Monte-Carlo...\n")
    println("Neural network regularization parameter: $(preTrainParms.PTREGP)")
    reportOpt(opt)

    rng = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    nsystems = length(systemParmsList)

    # Pre-compute reference data in parallelrefRDFs
    refDataInputs = []
    for systemId in 1:nsystems
        refDataInput = PreComputedInput(NNParms, systemParmsList[systemId], refRDFs[systemId])
        append!(refDataInputs, [refDataInput])
    end
    refDataList::Vector{ReferenceData} = pmap(preComputeRefData, refDataInputs)

    for step in 1:(preTrainParms.PTsteps)
        verbose::Bool = false
        if step % preTrainParms.PToutfreq == 0 || step == 1
            verbose = true
            println("\nStep $(step)...\n")
        end
        lossGradients = []
        for systemId in 1:nsystems
            # Pack frame input arrays
            refData = refDataList[systemId]

            # Run a pre-training step, compute energy differences with PMF and the neural network,
            # restore all input arguments to their original state
            symmFuncMatrix1, symmFuncMatrix2, ΔENN, ΔEPMF = pretraining_move!(refData, model, NNParms,
                                                                              systemParmsList[systemId], rng)

            # Compute the loss gradient
            lossGradient = computePreTrainingLossGradients(ΔENN, ΔEPMF, symmFuncMatrix1, symmFuncMatrix2, model,
                                                           preTrainParms, NNParms, verbose)
            append!(lossGradients, [lossGradient])
        end
        # Update the model
        meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        updatemodel!(model, opt, meanLossGradients)
    end
    @save "model-pre-trained.bson" model
    checkfile("model-pre-trained.bson")
    return (model)
end

function main()
    # Start the timer
    startTime = Dates.now()
    println("Starting at: ", startTime)

    # Initialize the parameters
    globalParms, MCParms, NNParms, preTrainParms, systemParmsList = parametersInit()

    # Check if the number of workers is divisble by the number of ref systems
    num_workers = nworkers()
    num_systems = length(systemParmsList)
    @assert(num_workers % num_systems==0,
            "Number of requested CPU cores ($num_workers) "*"must be divisible by the number of systems ($num_systems)!")

    # Initialize the input data
    inputs = inputInit(globalParms, NNParms, preTrainParms, systemParmsList)

    if globalParms.mode == "training"
        model, opt, refRDFs = inputs
    else
        model = inputs
    end

    # Print information about symmetry functions
    println("Using the following symmetry functions as the neural input for each atom:")
    print_symmetry_function_info(NNParms)

    if globalParms.mode == "training"
        println("Training a model using $(num_systems) reference system(s)")
        println("Using the following activation functions: $(NNParms.activations)")
        if globalParms.modelFile == "none"
            # Run pretraining
            model = pretrain!(preTrainParms, NNParms, systemParmsList, model, opt, refRDFs)
            # Restore optimizer state to default
            println("\nRe-initializing the optimizer for the training...\n")
            opt = optInit(NNParms)
            reportOpt(opt)
            println("Neural network regularization parameter: $(NNParms.REGP)")
        end
        # Run the training
        println("\nStarting the main part of the training...\n")
        println("Adaptive gradient scaling: $(globalParms.adaptiveScaling)")
        println("Number of iterations: $(NNParms.iters)")
        println("Running MC simulation on $(num_workers) rank(s)...\n")
        println("Total number of steps: $(MCParms.steps * num_workers / 1E6)M")
        println("Number of equilibration steps per rank: $(MCParms.Eqsteps / 1E6)M")
        train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    else
        @assert(length(systemParmsList)==1, "Only one system at a time can be simulated!")
        println("Running simulation with $(globalParms.modelFile)")
        # Run the simulation
        simulate!(model, globalParms, MCParms, NNParms, systemParmsList[1])
    end

    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    return println("Walltime: ", wallTime)
end

end # module MlImc

if abspath(PROGRAM_FILE) == @__FILE__
    MlImc.main()
end
