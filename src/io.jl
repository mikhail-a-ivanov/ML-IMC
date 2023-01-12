using Printf
using Chemfiles

"""
struct globalParameters

Fields:
systemFiles: 
    list of input filenames for each system
mode: 
    ML-IMC mode - training with reference data or simulation using a trained model
inputmodel: 
    "random" keyword for random initialization,
    "zero" for zeros in the first layer or a filename of a trained model
outputMode:
    default (rdf, energy, model); verbose (+gradients, +trajectories)            
"""
struct globalParameters
    systemFiles::Vector{String}
    mode::String
    inputmodel::String
    outputMode::String
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
struct NNparameters

Fields:
preTrainSteps: number of pre-training steps
preTrainOutFreq: frequency of pre-training reporting
minR: min distance for G2 symmetry function, Å
maxR: max distance for G2 symmetry function (cutoff), Å 
sigma: sigma parameter in G2 symmetry function (gaussian width), Å
neurons: number of neurons in the network
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
    preTrainSteps::Int64
    preTrainOutFreq::Int64
    minR::Float64
    maxR::Float64
    sigma::Float64
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
struct systemParameters

Fields:
systemName: name of the system
topname: name of the topology file
trajfile: name of the trajectory file
N: number of particles
atomname: atomic symbol
box: box vector, Å
V: volume, Å^3
rdfname: reference RDF file
Nbins: number of histogram bins
binWidth: histogram bin width, Å
repulsionLimit: minimum allowed pair distance, Å
T: temperature, K
β: 1/(kB*T), reciprocal kJ/mol
Δ: max displacement, Å
targetAR: target acceptance ratio
"""
struct systemParameters
    systemName::String
    trajfile::String
    topname::String
    N::Int
    atomname::String
    box::Vector{Float64}
    V::Float64
    rdfname::String
    Nbins::Int
    binWidth::Float64
    repulsionLimit::Float64
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
    file = open(inputname, "r")
    lines = readlines(file)
    splittedLines = [split(line) for line in lines]

    # Make a list of field names
    globalFields = [String(field) for field in fieldnames(globalParameters)]
    MCFields = [String(field) for field in fieldnames(MCparameters)]
    NNFields = [String(field) for field in fieldnames(NNparameters)]

    # Input variable arrays
    globalVars = []
    MCVars = []
    NNVars = []

    # Loop over fieldnames and fieldtypes and over splitted lines
    # Global parameters
    for (field, fieldtype) in zip(globalFields, fieldtypes(globalParameters))
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
    globalParms = globalParameters(globalVars...)

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
                    neurons = []
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
    NNParms = NNparameters(NNVars...)

    # Read system input files
    systemParmsList = [] # list of systemParameters structs
    systemFields = [String(field) for field in fieldnames(systemParameters)]
    for inputname in globalParms.systemFiles
        systemVars = []
        file = open(inputname, "r")
        lines = readlines(file)
        splittedLines = [split(line) for line in lines]
        for (field, fieldtype) in zip(systemFields, fieldtypes(systemParameters))
            for line in splittedLines
                if length(line) != 0 && field == line[1]
                    if field == "T"
                        T = parse(fieldtype, line[3])
                        β = 1 / (kB * T)
                        append!(systemVars, T)
                        append!(systemVars, β)
                    elseif field == "topname"
                        topname = [line[3]]
                        pdb = Trajectory("$(topname[1])")
                        pdbFrame = read(pdb)
                        N = length(pdbFrame)
                        atomname = name(Atom(pdbFrame, 1))
                        box = lengths(UnitCell(pdbFrame))
                        V = box[1] * box[2] * box[3]
                        append!(systemVars, topname)
                        append!(systemVars, N)
                        append!(systemVars, [atomname])
                        append!(systemVars, [box])
                        append!(systemVars, V)
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
        systemParms = systemParameters(systemVars...)
        append!(systemParmsList, [systemParms])
    end

    if globalParms.mode == "training"
        println("Running ML-IMC in the training mode.")
    else
        println("Running ML-IMC in the simulation mode.")
    end

    return (globalParms, MCParms, NNParms, systemParmsList)
end

"""
function readXTC(systemParms)

Reads input configurations from XTC file
"""
function readXTC(systemParms)
    traj = Trajectory(systemParms.trajfile)
    return (traj)
end

"""
function inputInit(globalParms, NNParms, systemParmsList)

Initializes input data
"""
function inputInit(globalParms, NNParms, systemParmsList)
    # Read reference data
    refRDFs = []
    for systemParms in systemParmsList
        bins, refRDF = readRDF(systemParms.rdfname)
        append!(refRDFs, [refRDF])
    end

    # Set up a model and an optimizer for training
    # or load a model from a file for MC sampling

    # Initialize the optimizer
    opt = optInit(NNParms)
    if globalParms.inputmodel == "random" || globalParms.inputmodel == "zero"
        # Initialize the model
        model = modelInit(NNParms, globalParms)
    else
        @load globalParms.inputmodel model
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
end

"""
function writetraj(conf, parameters, outname, mode='w')

Writes a wrapped configuration into a trajectory file (Depends on Chemfiles)
"""
function writeTraj(conf, systemParms, outname, mode='w')
    # Create an empty Frame object
    frame = Frame()
    # Set PBC vectors
    boxCenter = systemParms.box ./ 2
    set_cell!(frame, UnitCell(systemParms.box))
    # Add wrapped atomic coordinates to the frame
    for i = 1:systemParms.N
        wrappedAtomCoords = wrap!(UnitCell(frame), conf[:, i]) .+ boxCenter
        add_atom!(frame, Atom(systemParms.atomname), wrappedAtomCoords)
    end
    # Write to file
    Trajectory(outname, mode) do traj
        write(traj, frame)
    end
    return
end

"""
function readRDF(rdfname)
Reads RDF produced by mcLJ.jl
"""
function readRDF(rdfname)
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
