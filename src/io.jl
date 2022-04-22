using BSON: @save, @load

include("readLJ.jl")

"""
mutable struct inputParms

Fields:
N: number of particles
box: box vector, Å
V: volume, Å^3
T: temperature, K
β: 1/(kB*T), reciprocal kJ/mol
delta: max displacement, Å
steps: total number of steps
Eqsteps: equilibration steps
xyzout: XYZ output frequency
outfreq: output frequency
binWidth: histogram bin width, Å
Nbins: number of histogram bins
iters: number of learning iterations
activation: activation function
optimizer: type of optimizer
rate: learning rate
rateAdjust: learning rate multiplier
momentum: momentum coefficient
xyzname: input configuration file
rdfname: reference RDF file
paircorr: type of pair correlations (RDF or histogram)
neurons: number of neurons in the hidden layers
paramsInit: type of network parameters initialization
shift: shift parameter in the repulsion guess (stiffness*[exp(-alpha*r)-shift])
stiffness: stiffness parameter in the repulsion guess (stiffness*[exp(-alpha*r)-shift])
modelname: name of the trained model file
mode: ML-IMC mode: training with reference data or simulation using a trained model
"""
mutable struct inputParms
    N::Int
    box::SVector{3, Float32}
    V::Float32
    T::Float64
    β::Float64
    delta::Float32  
    steps::Int
    Eqsteps::Int
    xyzout::Int 
    outfreq::Int
    binWidth::Float32
    Nbins::Int
    iters::Int
    activation::String
    optimizer::String
    rate::Float64
    rateAdjust::Float64
    momentum::Float64
    xyzname::String
    rdfname::String
    paircorr::String
    neurons::Vector{Int}
    paramsInit::String
    shift::Float32
    stiffness::Float32
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
    NA::Float64 = 6.02214076E23 # [mol-1]
    kB::Float64 = 1.38064852E-23 * NA / 1000 # [kJ/(mol*K)]

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
                if field == "box"
                    box = zeros(3)
                    box[1] = parse(Float32, line[3])
                    box[2] = parse(Float32, line[4])
                    box[3] = parse(Float32, line[5])
                    append!(vars, [box])
                    V = prod(box)
                    append!(vars, V)       
                elseif field == "T"
                    T = parse(Float64, line[3])
                    β = 1/(kB * T)
                    append!(vars, T)  
                    append!(vars, β)
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
    return(parameters)
end

"""
function confsInit(parameters)

Reads input configurations from XYZ file
"""
function confsInit(parameters)
    xyz = readXYZ(parameters.xyzname)
    confs = xyz[2:end] # Omit the initial configuration
    @assert parameters.N == length(confs[end]) "Given number of particles does not match with XYZ configuration!"
    return(confs)
end

"""
function inputInit(parameters)

Initializes input data
"""
function inputInit(parameters)
    # Read reference histogram
    bins, rdfref, histref = readRDF(parameters.rdfname)

    # Allocate a vector for the reference descriptor data
    descriptorref = zeros(Float32, length(bins))

    if parameters.paircorr == "RDF"
        descriptorref = rdfref
    elseif parameters.paircorr == "histogram"
        # Normalize the reference histogram to per particle histogram
        histref ./= parameters.N / 2 # Number of pairs divided by the number of particles
        descriptorref = histref
    end

    # Read XYZ configurations
    confs = confsInit(parameters)

    if parameters.mode == "training"
        # Initialize the optimizer
        opt = optInit(parameters)
        # Make a copy to read at the start of each iterations
        refconfs = copy(confs)
        # Initialize the model
        model = modelInit(descriptorref, parameters)
    else
        @load parameters.modelname model
    end

    # Initialize RNG for random input frame selection
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    if parameters.mode == "training"
        return(confs, model, opt, refconfs, descriptorref, rng_xor)
    else
        return(confs, model, rng_xor)
    end
end

"""
function writedescriptor(outname, descriptor, parameters)

Writes the descriptor into a file
"""
function writedescriptor(outname, descriptor, parameters)
    bins = [bin*parameters.binWidth for bin in 1:parameters.Nbins]
    io = open(outname, "w")
    if parameters.paircorr == "RDF"
        print(io, "# r, Å; RDF \n")
    elseif parameters.paircorr == "histogram"
        print(io, "# r, Å; Distance histogram \n")
    end
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