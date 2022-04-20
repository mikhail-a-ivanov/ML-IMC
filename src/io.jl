"""
mutable struct inputParms

Fields:
N: number of particles
box: box vector, Å
T: temperature, K
beta: 1/(kB*T), reciprocal kJ/mol
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
momentum: momentum coefficient
xyzname: input configuration file
rdfname: reference RDF file
paircorr: type of pair correlations (RDF or histogram)
neurons: number of neurons in the hidden layers
paramsInit: type of network parameters initialization
"""
mutable struct inputParms
    N::Int
    box::SVector{3, Float64}
    T::Float64
    beta::Float64
    delta::Float64  
    steps::Int
    Eqsteps::Int
    xyzout::Int 
    outfreq::Int
    binWidth::Float64
    Nbins::Int
    iters::Int
    activation::String
    optimizer::String
    rate::Float64
    momentum::Float64
    xyzname::String
    rdfname::String
    paircorr::String
    neurons::Vector{Int64}
    paramsInit::String
end

"""
readinput(inputname)

Reads an input file for ML-IMC
and saves the data into the
inputParms struct
"""
function readinput(inputname)
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
                    box[1] = parse(Float64, line[3])
                    box[2] = parse(Float64, line[4])
                    box[3] = parse(Float64, line[5])
                    append!(vars, [box])       
                elseif field == "T"
                    T = parse(Float64, line[3])
                    beta = 1/(kB * T)
                    append!(vars, T)  
                    append!(vars, beta)
                elseif field == "neurons"
                    neurons = []
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(neurons, parse(Int64, strip(element, ',')))
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
function writedescriptor(outname, descriptor, parameters)

Writes the descriptor into a file
"""
function writedescriptor(outname, descriptor, bins)
    io = open(outname, "w")
    print(io, "# r, Å; Descriptor \n")
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
function savemodel(outname, model)

Saves model into a file
"""
function savemodel(outname, model)
    io = open(outname, "w")
    print(io, "# ML-IMC model: $(length(model.weight)) weights; $(length(model.bias)) biases\n")
    print(io, "# Weights\n")
    for weight in model.weight
        print(io, @sprintf("%12.8f", weight), "\n")
    end
    print(io, "# Biases\n")
    for bias in model.bias
        print(io, @sprintf("%12.8f", bias), "\n")
    end
    close(io)
end