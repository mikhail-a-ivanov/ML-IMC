"""
struct inputParms

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
"""
struct inputParms
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
end


"""
readXYZ(xyzname)

Reads an XYZ file and outputs coordinates of all atoms
arranged in separate frames
"""
function readXYZ(xyzname)
    file = open(xyzname, "r")
    lines = readlines(file)
    natoms = parse(Int, lines[1])
    ncomments = 2
    nlines = length(lines)
    nframes = Int((nlines % natoms) / 2)
    xyz = [[zeros(3) for i in 1:natoms] for j in 1:nframes]
    println("\nReading $(xyzname) with $(nlines) lines...")
    println("Found $(nframes) frames with $(natoms) atoms each...")
    for lineId in 1:nlines
        frameId = ceil(Int, lineId/(natoms + ncomments))
        atomId = lineId - ((frameId - 1) * (natoms + ncomments)) - ncomments
        @assert atomId <= natoms
        if atomId > 0
            atomLine = split(lines[lineId])
            for i in 1:3
                xyz[frameId][atomId][i] = parse(Float64, atomLine[i+1])
            end
        end
    end
    close(file)
    println("Success! Closing the file...")
    return(xyz)
end

"""
readRDF(rdfname)

Reads RDF and distance histogram produced
by mcLJ.jl
"""
function readRDF(rdfname)
    file = open(rdfname, "r")
    println("\nReading reference data from $(rdfname)...")
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
    return(bins, rdf, hist)
    close(file)
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
                    #println("$(field) = $(box)")            
                elseif field == "T"
                    T = parse(Float64, line[3])
                    beta = 1/(kB * T)
                    append!(vars, T)  
                    append!(vars, beta)
                    #println("T = $(T)")
                    #println("beta = $(beta)")
                else
                    #println("$(field) = $(line[3])")
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