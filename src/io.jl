using BSON: @save, @load

"""
mutable struct inputParms

Fields:
systemFiles: list of input filenames for each system
steps: total number of steps
Eqsteps: equilibration steps
stepAdjustFreq: frequency of MC step adjustment
targetAR: target acceptance ratio
outfreq: output frequency
trajout: trajectory output frequency
iters: number of learning iterations
activation: activation function
optimizer: type of optimizer
rate: learning rate
momentum: momentum coefficient
neurons: number of neurons in the hidden layers
modelname: name of the trained model file
mode: ML-IMC mode: training with reference data or simulation using a trained model
"""
mutable struct inputParms
    systemFiles::Vector{String}
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    outfreq::Int
    trajout::Int
    iters::Int
    activation::String
    optimizer::String
    rate::Float64
    momentum::Float64
    neurons::Vector{Int}
    mode::String
    modelname::String
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
delta: max displacement, Å
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
    delta::Float64
    targetAR::Float64
end

"""
parametersInit()

Reads an input file for ML-IMC
and saves the data into the
inputParms struct
"""
function parametersInit()
    # Read the input name from the command line argument
    inputname = ARGS[1]

    # Constants
    NA::Float64 = 6.02214076 # [mol-1] * 10^-23
    kB::Float64 = 1.38064852 * NA / 1000 # [kJ/(mol*K)]

    # Read the input file
    file = open(inputname, "r")
    lines = readlines(file)
    splittedLines = [split(line) for line in lines]

    # Make a list of field names
    fields = [String(field) for field in fieldnames(inputParms)]

    vars = [] # Array with input variables
    # Loop over fieldnames and fieldtypes and over splitted lines
    for (field, fieldtype) in zip(fields, fieldtypes(inputParms))
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
                    append!(vars, [systemFiles])
                elseif field == "neurons"
                    neurons = []
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(neurons, parse(Int, strip(element, ',')))
                        elseif element == "#"
                            break
                        end
                    end
                    append!(vars, [neurons])
                elseif field == "mode"
                    mode = [line[3]]
                    append!(vars, mode) 
                    if mode[1] == "training"
                        modelname = " "
                        append!(vars, [modelname])
                    end
                else
                    if fieldtype != String
                        append!(vars, parse(fieldtype, line[3]))
                    else
                        append!(vars, [line[3]])
                    end
                end
            end
        end
    end

    # Save parameters into the inputParms struct
    parameters = inputParms(vars...)
    if parameters.mode == "training"
        println("Running ML-IMC in the training mode.\n")
    else
        println("Running ML-IMC in the simulation mode.\n") 
    end

    # Read system input files
    systemParmsList = [] # list of systemParameters structs
    systemFields = [String(field) for field in fieldnames(systemParameters)]
    for inputname in parameters.systemFiles
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

    return(parameters, systemParmsList)
end

"""
function inputInit(parameters, systemParmsList)

Initializes input data
"""
function inputInit(parameters, systemParmsList)
    refRDFs = []
    # Read reference RDFs
    for systemParms in systemParmsList
        bins, rdfref = readRDF(systemParms.rdfname)
        append!(refRDFs, [rdfref])
    end

    if parameters.mode == "training"
        # Initialize the optimizer
        opt = optInit(parameters)
        # Initialize the model
        model = modelInit(parameters)
    else
        @load parameters.modelname model
    end

    if parameters.mode == "training"
        return(model, opt, refRDFs)
    else
        return(model)
    end
end

"""
function writedescriptor(outname, descriptor, systemParms)

Writes the descriptor into a file
"""
function writedescriptor(outname, descriptor, systemParms)
    bins = [bin*systemParms.binWidth for bin in 1:systemParms.Nbins]
    io = open(outname, "w")
    print(io, "# r, Å; RDF \n")
    print(io, @sprintf("%6.3f %12.3f", bins[1], 0), "\n")
    for i in 2:length(descriptor)
        print(io, @sprintf("%6.3f %12.3f", bins[i], descriptor[i]), "\n")
    end
    close(io)
end

"""
function writeenergies(outname, energies)

Writes the total energy to an output file
"""
function writeenergies(outname, energies, parameters, slicing=10)
    steps = 0:parameters.outfreq*slicing:parameters.steps
    io = open(outname, "w")
    print(io, @sprintf("%8s %22s", "# Step", "Total energy, kJ/mol"))
    print(io, "\n")
    for i in 1:length(energies[1:slicing:end])
        print(io, @sprintf("%9d %10.3f", steps[i], energies[1:slicing:end][i]), "\n")
    end
    close(io)
end

"""
function writetraj(conf, systemParms, outname, mode='w')
Writes a wrapped configuration into a trajectory file (Depends on Chemfiles)
"""
function writetraj(conf, systemParms, outname, mode='w')
    # Create an empty Frame object
    frame = Frame() 
    # Set PBC vectors
    boxCenter = systemParms.box ./ 2
    set_cell!(frame, UnitCell(systemParms.box))
    # Add wrapped atomic coordinates to the frame
    for i in 1:systemParms.N
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
readRDF(rdfname)

Reads RDF produced by mcLJ.jl
"""
function readRDF(rdfname)
    file = open(rdfname, "r")
    lines = readlines(file)
    ncomments = 2
    nlines = length(lines) - ncomments
    bins = zeros(nlines)
    rdf = zeros(nlines)
    for i in (1 + ncomments):length(lines)
        rdfline = split(lines[i])
        if rdfline[1] != "#"
            bins[i - ncomments] = parse(Float32, rdfline[1])
            rdf[i - ncomments] = parse(Float32, rdfline[2])
        end
    end
    return(bins, rdf)
    close(file)
end