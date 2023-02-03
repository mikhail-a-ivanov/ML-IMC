using Printf
using Chemfiles
using BSON: @save, @load

"""
struct globalParameters

Fields:
systemFiles: 
    list of input filenames for each system
symmetryFunctionFile:
    name of file containing symmetry function information
mode: 
    ML-IMC mode - training with reference data or simulation using a trained model
outputMode:
    default (rdf, energy, model, opt, gradients); verbose (+trajectories)     
modelFile: 
    "none" keyword for random initialization
    name of model file to do a restart
gradientsFile:
    "none" keyword for default initialization
    name of gradients file to do a restart (couple with a corresponding model/opt filenames)
optimizerFile:
    "none" keyword for default initialization
    name of optimizer file to do a restart (couple with a corresponding model/gradients filenames)
"""
struct GlobalParameters
    systemFiles::Vector{String}
    symmetryFunctionFile::String
    mode::String
    outputMode::String
    modelFile::String
    gradientsFile::String
    optimizerFile::String
end

"""
struct MCparameters

Fields:
steps: total number of steps
Eqsteps: equilibration steps
stepAdjustFreq: frequency of MC step adjustment
trajout: XTC trajectory output frequency
outfreq: output frequency
"""
struct MCparameters
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    trajout::Int
    outfreq::Int
end

"""
struct G2

Radial symmetry function:
G2 = ∑ exp(-eta * (Rij - rshift))^2 * fc(Rij, rcutoff)
where fc is the smooth cutoff function  

Fields:
eta (Å^-2): 
    parameter controlling the width of G2
    eta = 1 / sqrt(2 * sigma)
rcutoff (Å):
    Cutoff radius
rshift (Å):
    distance shifting parameter
"""
struct G2
    eta::Float64
    rcutoff::Float64
    rshift::Float64
end

"""
struct G3

Narrow angular symmetry function
G3 = 2^(1 - zeta) * ∑ (1 + lambda * cos(theta))^zeta * 
    exp(-eta * ((Rij - rshift)^2 + (Rik - rshift)^2 + (Rkj - rshift)^2) * 
    fc(Rij, rcutoff) * fc(Rik, rcutoff) * fc(Rkj, rcutoff)

where fc is the smooth cutoff function  

Fields:
eta (Å^-2): 
    parameter controlling the width of the radial part of G9
    eta = 1 / sqrt(2 * sigma)
lambda: 
    parameter controlling the phase of cosine function
    either +1 or -1
zeta:
    parameter controlling angular resolution
rcutoff (Å):
    Cutoff radius
rshift (Å):
    distance shifting parameter
"""
struct G3
    eta::Float64
    lambda::Float64
    zeta::Float64
    rcutoff::Float64
    rshift::Float64
end


"""
struct G9

Wide angular symmetry function
G9 = 2^(1 - zeta) * ∑ (1 + lambda * cos(theta))^zeta * 
    exp(-eta * ((Rij - rshift)^2 + (Rik - rshift)^2) * 
    fc(Rij, rcutoff) * fc(Rik, rcutoff)

where fc is the smooth cutoff function  

Fields:
eta (Å^-2): 
    parameter controlling the width of the radial part of G9
    eta = 1 / sqrt(2 * sigma)
lambda: 
    parameter controlling the phase of cosine function
    either +1 or -1
zeta:
    parameter controlling angular resolution
rcutoff (Å):
    Cutoff radius
rshift (Å):
    distance shifting parameter
"""
struct G9
    eta::Float64
    lambda::Float64
    zeta::Float64
    rcutoff::Float64
    rshift::Float64
end

"""
struct NNparameters

Fields:
G2Functions: list of G2 symmetry function parameters
G3Functions: list of G3 symmetry function parameters  
G9Functions: list of G9 symmetry function parameters
maxDistanceCutoff: max distance cutoff
symmFunctionScaling: scaling factor for the symmetry functions
neurons: number of hidden neurons in the network
iters: number of learning iterations
activations: list of activation functions
REGP: regularization parameter
optimizer: type of optimizer
rate: learning rate
momentum: momentum coefficient
decay1: decay of the optimizer (1)
decay2: decay of the optimizer (2)
"""
struct NNparameters
    G2Functions::Vector{G2}
    G3Functions::Vector{G3}
    G9Functions::Vector{G9}
    maxDistanceCutoff::Float64
    symmFunctionScaling::Float64
    neurons::Vector{Int}
    iters::Int
    activations::Vector{String}
    REGP::Float64
    optimizer::String
    rate::Float64
    momentum::Float64
    decay1::Float64
    decay2::Float64
end

"""
struct preTrainParameters

Fields:
PTsteps: number of pre-training steps
PToutfreq: frequency of pre-training reporting
The rest as in NNparameters but with PT prefix
"""
struct PreTrainParameters
    PTsteps::Int64
    PToutfreq::Int64
    PTREGP::Float64
    PToptimizer::String
    PTrate::Float64
    PTmomentum::Float64
    PTdecay1::Float64
    PTdecay2::Float64
end

"""
struct systemParameters

Fields:
systemName: name of the system
topname: name of the topology file
trajfile: name of the trajectory file
N: number of particles
atomname: atomic symbol
rdfname: reference RDF file
Nbins: number of histogram bins
binWidth: histogram bin width, Å
T: temperature, K
β: 1/(kB*T), reciprocal kJ/mol
Δ: max displacement, Å
targetAR: target acceptance ratio
"""
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

"""
function parametersInit()

Reads an input file for ML-IMC
and saves the data into
parameter structs
"""
function parametersInit()
    # Read the input name from the command line argument
    #inputname = ARGS[1]
    inputname = "ML-IMC-init.in"

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
    MCFields = [String(field) for field in fieldnames(MCparameters)]
    NNFields = [String(field) for field in fieldnames(NNparameters)]
    preTrainFields = [String(field) for field in fieldnames(PreTrainParameters)]

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
                append!(G2Parameters, parse(fieldtype, line[fieldIndex+1]))
            end
            append!(G2s, [G2(G2Parameters...)])
        end
        if length(line) != 0 && line[1] == "G3"
            G3Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G3))
                append!(G3Parameters, parse(fieldtype, line[fieldIndex+1]))
            end
            append!(G3s, [G3(G3Parameters...)])
        end
        if length(line) != 0 && line[1] == "G9"
            G9Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G9))
                append!(G9Parameters, parse(fieldtype, line[fieldIndex+1]))
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
    for (field, fieldtype) in zip(MCFields, fieldtypes(MCparameters))
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
    MCParms = MCparameters(MCVars...)

    # Loop over fieldnames and fieldtypes and over splitted lines
    for (field, fieldtype) in zip(NNFields, fieldtypes(NNparameters))
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
    NNParms = NNparameters(G2s, G3s, G9s, maxDistanceCutoff, scalingFactor, NNVars...)

    # Pre-training parameters
    for (field, fieldtype) in zip(preTrainFields, fieldtypes(PreTrainParameters))
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
    preTrainParms = PreTrainParameters(preTrainVars...)

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

"""
function readXTC(systemParms)

Reads input configurations from XTC file
"""
function readXTC(systemParms)
    checkfile(systemParms.trajfile)
    traj = Trajectory(systemParms.trajfile)
    return (traj)
end

"""
function inputInit(globalParms, NNParms, preTrainParms, systemParmsList)

Initializes input data
"""
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

    if globalParms.mode == "training"
        return (model, opt, refRDFs)
    else
        return (model)
    end
end

"""
function writeRDF(outname, rdf, systemParms)

Writes RDF into a file
"""
function writeRDF(outname, rdf, systemParms)
    bins = [bin * systemParms.binWidth for bin = 1:systemParms.Nbins]
    # Write the data
    io = open(outname, "w")
    print(io, "# System: $(systemParms.systemName)\n")
    print(io, "# RDF data ($(systemParms.atomname) - $(systemParms.atomname)) \n")
    print(io, "# r, Å; g(r); \n")
    for i in eachindex(rdf)
        print(io, @sprintf("%6.3f %12.3f", bins[i], rdf[i]), "\n")
    end
    close(io)
    checkfile(outname)
end

"""
function writeenergies(outname, energies, MCParms, systemParms, slicing=1)

Writes the total energy to an output file
"""
function writeEnergies(outname, energies, MCParms, systemParms, slicing=1)
    steps = 0:MCParms.outfreq*slicing:MCParms.steps
    io = open(outname, "w")
    print(io, "# System: $(systemParms.systemName)\n#")
    print(io, @sprintf("%8s %22s", " Step", "Total energy, kJ/mol"))
    print(io, "\n")
    for i = 1:length(energies[1:slicing:end])
        print(io, @sprintf("%9d %10.3f", steps[i], energies[1:slicing:end][i]), "\n")
    end
    close(io)
    checkfile(outname)
end

"""
function writeTraj(conf, box, parameters, outname, mode='w')

Writes a wrapped configuration into a trajectory file (Depends on Chemfiles)
"""
function writeTraj(conf, box, systemParms, outname, mode='w')
    # Create an empty Frame object
    frame = Frame()
    # Set PBC vectors
    boxCenter = box ./ 2
    set_cell!(frame, UnitCell(box))
    # Add wrapped atomic coordinates to the frame
    for i = 1:systemParms.N
        wrappedAtomCoords = wrap!(UnitCell(frame), conf[:, i]) .+ boxCenter
        add_atom!(frame, Atom(systemParms.atomname), wrappedAtomCoords)
    end
    # Write to file
    Trajectory(outname, mode) do traj
        write(traj, frame)
    end
    checkfile(outname)
    return
end

"""
function readRDF(rdfname)
Reads RDF produced by mcLJ.jl
"""
function readRDF(rdfname)
    checkfile(rdfname)
    file = open(rdfname, "r")
    lines = readlines(file)
    ncomments = 2
    nlines = length(lines) - ncomments
    bins = zeros(nlines)
    rdf = zeros(nlines)
    for i = (1+ncomments):length(lines)
        rdfline = split(lines[i])
        if rdfline[1] != "#"
            bins[i-ncomments] = parse(Float64, rdfline[1])
            rdf[i-ncomments] = parse(Float64, rdfline[2])
        end
    end
    return (bins, rdf)
    close(file)
end

function checkfile(filename)
    @assert isfile(filename) "Could not locate $(filename)!"
end
