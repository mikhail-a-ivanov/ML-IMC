using Chemfiles
using BSON: @save, @load

"""
struct globalParameters

Fields:
systemFiles: 
    list of input filenames for each system
mode: 
    ML-IMC mode - training with reference data or simulation using a trained model
modelname: 
    name of the trained model file
descent: 
    unrestrticted - mean loss can increase during training
    restricted - decreases learning rate if the mean loss increases, proceeds 
    to the next iteration only when the mean loss decreases 
"""
struct globalParameters
    systemFiles::Vector{String}
    mode::String
    modelname::String
    descent::String
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
neurons: number of neurons in the network (excluding the energy output neuron)
iters: number of learning iterations
activation: activation function
optimizer: type of optimizer
rate: learning rate
rateAdjust: learning rate multiplier
μ: momentum coefficient
minR: min distance for G2 symmetry function, Å
maxR: max distance for G2 symmetry function (cutoff), Å 
η: η parameter in G2 symmetry function (gaussian width), Å
"""
mutable struct NNparameters
    neurons::Vector{Int}
    iters::Int
    activation::String
    optimizer::String
    rate::Float64
    rateAdjust::Float64
    μ::Float64
    minR::Float64
    maxR::Float64
    η::Float64
end

"""
mutable struct systemParameters

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
repulsionLimit: minimum allowed pair distance (at which RDF > 0), Å
T: temperature, K
β: 1/(kB*T), reciprocal kJ/mol
Δ: max displacement, Å
targetAR: target acceptance ratio
"""
mutable struct systemParameters
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
parametersInit()

Reads an input file for ML-IMC
and saves the data into
parameter structs
"""
function parametersInit()
    # Read the input name from the command line argument
    inputname = ARGS[1]
    #inputname = "ML-IMC-init.in"

    # Constants
    NA::Float64 = 6.02214076E23 # [mol-1]
    kB::Float64 = 1.38064852E-23 * NA / 1000 # [kJ/(mol*K)]

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
                elseif field == "mode"
                    mode = [line[3]]
                    append!(globalVars, mode) 
                    if mode[1] == "training"
                        modelname = " "
                        append!(globalVars, [modelname])
                    end
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
                        β = 1/(kB * T)
                        append!(systemVars, T)  
                        append!(systemVars, β)
                    elseif field == "topname"
                        topname = [line[3]]
                        pdb = Trajectory("$(topname[1])")
                        pdb_frame = read(pdb)
                        N = length(pdb_frame)
                        atomname = name(Atom(pdb_frame, 1))
                        box = lengths(UnitCell(pdb_frame))
                        V = box[1] * box[2] * box[3]
                        append!(systemVars, topname)
                        append!(systemVars, N)
                        append!(systemVars, [atomname])
                        append!(systemVars, [box])
                        append!(systemVars, V)
                    elseif field == "rdfname"
                        rdfname = [line[3]]
                        bins, rdf, hist, repulsionLimit = readRDF("$(rdfname[1])")
                        Nbins = length(bins)
                        binWidth = bins[1]
                        append!(systemVars, [rdfname[1]])
                        append!(systemVars, Nbins)
                        append!(systemVars, binWidth)
                        append!(systemVars, repulsionLimit)
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

    return(globalParms, MCParms, NNParms, systemParmsList)
end

"""
readXTC(systemParms)

Reads input configurations from XTC file
"""
function readXTC(systemParms)
    traj = Trajectory(systemParms.trajfile)
    return(traj)
end

"""
inputInit(globalParms, NNParms, systemParmsList)

Initializes input data
"""
function inputInit(globalParms, NNParms, systemParmsList)
    # Read reference data
    refRDFs = []
    trajectories = []
    for systemParms in systemParmsList
        bins, refRDF, refhist, repulsionLimit = readRDF(systemParms.rdfname)
        append!(refRDFs, [refRDF])
    end

    # Set up a model and an optimizer for training
    # or load a model from a file for MC sampling
    if globalParms.mode == "training"
        # Initialize the optimizer
        opt = optInit(NNParms)
        # Initialize the model
        model = modelInit(NNParms)
    else
        @load globalParms.modelname model
    end

    if globalParms.mode == "training"
        return(model, opt, refRDFs)
    else
        return(model)
    end
end

"""
writeRDF(outname, rdf, systemParms)
Writes RDF into a file
"""
function writeRDF(outname, rdf, systemParms)
    bins = [bin*systemParms.binWidth for bin in 1:systemParms.Nbins]
    # Write the data
    io = open(outname, "w")
    print(io, "# System: $(systemParms.systemName)\n")
    print(io, "# RDF data ($(systemParms.atomname) - $(systemParms.atomname)) \n")
    print(io, "# r, Å; g(r); \n")
    for i in 1:length(rdf)
        print(io, @sprintf("%6.3f %12.3f", bins[i], rdf[i]), "\n")
    end
    close(io)
end

"""
writeenergies(outname, energies)

Writes the total energy to an output file
"""
function writeenergies(outname, energies, MCParms, slicing=1)
    steps = 0:MCParms.outfreq*slicing:MCParms.steps
    io = open(outname, "w")
    print(io, "# Total energy, kJ/mol \n")
    for i in 1:length(energies[1:slicing:end])
        print(io, "# Step = ", @sprintf("%d", steps[i]), "\n")
        print(io, @sprintf("%10.3f", energies[1:slicing:end][i]), "\n")
        print(io, "\n")
    end
    close(io)
end

"""
readRDF(rdfname)

Reads RDF and distance histogram produced
by mcLJ.jl
"""
function readRDF(rdfname)
    file = open(rdfname, "r")
    #println("Reading reference data from $(rdfname)...")
    lines = readlines(file)
    ncomments = 2
    nlines = length(lines) - ncomments
    bins = zeros(nlines)
    rdf = zeros(nlines)
    hist = zeros(nlines)
    for i in (1 + ncomments):length(lines)
        rdfline = split(lines[i])
        if length(rdfline) == 3
            bins[i - ncomments] = parse(Float64, rdfline[1])
            rdf[i - ncomments] = parse(Float64, rdfline[2])
            hist[i - ncomments] = parse(Float64, rdfline[3])
        end
    end
    # Find the repulsion limit
    repulsionLimit = 0. # Default value
    for i in 1:length(rdf)
        if rdf[i] > 0.
            repulsionLimit = bins[i - 1]
            break
        end
    end

    return(bins, rdf, hist, repulsionLimit)
    close(file)
end
