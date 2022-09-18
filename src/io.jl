using BSON: @save, @load

"""
mutable struct inputParms

Fields:
T: temperature, K
β: 1/(kB*T), reciprocal kJ/mol
delta: max displacement, Å
repulsionLimit: hard wall potential limit, Å
steps: total number of steps
Eqsteps: equilibration steps
stepAdjustFreq: frequency of MC step adjustment
targetAR: target acceptance ratio
xyzout: XYZ output frequency
outfreq: output frequency
iters: number of learning iterations
activation: activation function
optimizer: type of optimizer
rate: learning rate
momentum: momentum coefficient
topname: name of the topology file
N: number of particles
atomname: atomic symbol
box: box vector, Å
V: volume, Å^3
trajfile: name of the trajectory file
rdfname: reference RDF file
Nbins: number of histogram bins
binWidth: histogram bin width, Å
neurons: number of neurons in the hidden layers
modelname: name of the trained model file
mode: ML-IMC mode: training with reference data or simulation using a trained model
"""
mutable struct inputParms
    T::Float64
    β::Float64
    delta::Float32
    repulsionLimit::Float32  
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    targetAR::Float64
    xyzout::Int 
    outfreq::Int
    iters::Int
    activation::String
    optimizer::String
    rate::Float64
    momentum::Float64
    trajfile::String
    topname::String
    N::Int
    atomname::String
    box::Vector{Float32}
    V::Float32
    rdfname::String
    Nbins::Int
    binWidth::Float32
    neurons::Vector{Int}
    modelname::String
    mode::String
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
                if field == "T"
                    T = parse(Float64, line[3])
                    β = 1/(kB * T)
                    append!(vars, T)  
                    append!(vars, β)
                elseif field == "topname"
                    topname = [line[3]]
                    pdb = Trajectory("$(topname[1])")
                    pdb_frame = read(pdb)
                    N = length(pdb_frame)
                    atomname = name(Atom(pdb_frame, 1))
                    box = lengths(UnitCell(pdb_frame))
                    V = box[1] * box[2] * box[3]
                    append!(vars, topname)
                    append!(vars, N)
                    append!(vars, [atomname])
                    append!(vars, [box])
                    append!(vars, V)
                elseif field == "rdfname"
                    rdfname = [line[3]]
                    bins, rdf = readRDF("$(rdfname[1])")
                    Nbins = length(bins)
                    binWidth = bins[2] - bins[1]
                    append!(vars, [rdfname[1]])
                    append!(vars, Nbins)
                    append!(vars, binWidth)
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
    return(parameters)
end

"""
function inputInit(parameters)

Initializes input data
"""
function inputInit(parameters)
    # Read reference histogram
    bins, rdfref = readRDF(parameters.rdfname)

    if parameters.mode == "training"
        # Initialize the optimizer
        opt = optInit(parameters)
        # Initialize the model
        model = modelInit(parameters)
    else
        @load parameters.modelname model
    end

    if parameters.mode == "training"
        return(model, opt, rdfref)
    else
        return(model)
    end
end

"""
function writedescriptor(outname, descriptor, parameters)

Writes the descriptor into a file
"""
function writedescriptor(outname, descriptor, parameters)
    bins = [bin*parameters.binWidth for bin in 1:parameters.Nbins]
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