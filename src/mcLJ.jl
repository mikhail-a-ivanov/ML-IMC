using MPI
using Printf
using Dates
using RandomNumbers
using StaticArrays
using SharedArrays
using LinearAlgebra

"""
pbcdx(x1, x2, xsize)

Compute periodic boundary distance between x1 and x2
"""
function pbcdx(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int32, round(dx/xsize))
    return(dx)
end

"""
pbcdistance(p1, p2, box)

Compute 3D periodic boundary distance between points p1 and p2 
"""
function pbcdistance(p1, p2, box)
    R2::Float64 = 0.
    @fastmath @inbounds for i in 1:3
        R2 += pbcdx(p1[i], p2[i], box[i])^2
    end
    R = sqrt(R2)
    return(R)
end

"""
ljlattice(latticePoints, latticeScaling)

Generate a 3D latice of LJ atoms
separated by scaled Rm distance,
the periodic box vectors in reduced units
"""
function ljlattice(latticePoints, latticeScaling)
    lattice = [convert(SVector{3, Float64}, [i, j, k]) 
        for i in 0:latticePoints-1 for j in 0:latticePoints-1 for k in 0:latticePoints-1]
    scaling::Float64 = (2^(1/6)) * (latticeScaling)
    lattice = lattice .* scaling
    # Generate PBC box vectors
    boxSide::Float64 = (latticePoints) * (2^(1/6) * latticeScaling)
    box::SVector{3, Float64} = [boxSide, boxSide, boxSide]
    return(lattice, box)
end

"""
buildDistanceMatrix(conf, box)

Build distance matrix for a given
configuration
"""
function builddistanceMatrix(conf, box)
    # Build distance matrix
    distanceMatrix = zeros(Float64, length(conf), length(conf))
    @inbounds for i in 1:length(conf)
        for j in 1:length(conf)
            distanceMatrix[i,j] = pbcdistance(conf[i], conf[j], box)
        end
    end
    return(distanceMatrix)
end


"""
totalenergy(distanceMatrix)

Compute the total potential energy in reduced units
for a given distance matrix
"""
function totalenergy(distanceMatrix)
    N = convert(Int32, sqrt(length(distanceMatrix)))
    E::Float64 = 0.
    for i in 1:N
        for j in 1:i-1
            r6 = (1/distanceMatrix[i,j])^6
            E += 4 * (r6^2 - r6)
        end
    end
    return(E)
end

"""
particleenergy(distanceVector)

Computes the potential energy of one particle
from a given distance vector
"""
function particleenergy(distanceVector)
    E::Float64 = 0.
    @fastmath @inbounds for i in 1:length(distanceVector)
        if distanceVector[i] != 0
            r6 = (1/distanceVector[i])^6
            E += 4 * (r6^2 - r6)
        end
    end
    return(E)
end

"""
updatedistance(conf, box, distanceVector, pointIndex)

Updates distance vector
"""
function updatedistance!(conf, box, distanceVector, pointIndex)
    @fastmath @inbounds for i in 1:length(distanceVector)
        distanceVector[i] = pbcdistance(conf[i], conf[pointIndex], box)
    end
    return(distanceVector)
end

"""
hist!(distanceMatrix, hist, binWidth)

Computes RDF histogram
"""
function hist!(distanceMatrix, hist, binWidth)
    N = convert(Int32, sqrt(length(distanceMatrix)))
    for i in 1:N
        for j in 1:i-1
            histIndex = floor(Int32, 0.5 + distanceMatrix[i,j]/binWidth)
            if histIndex <= length(hist[1])
                hist[2][histIndex] += 1
            end
        end
    end
    return(hist)
end

"""
mcmove!(conf, box, distanceMatrix, E, beta, delta, rng)

Performs a Metropolis Monte Carlo
displacement move
"""
function mcmove!(conf, box, distanceMatrix, E, beta, delta, rng)
    # Pick a particle at random and calculate its energy
    pointIndex = rand(rng, Int32(1):Int32(length(conf)))
    distanceVector = distanceMatrix[:, pointIndex]
    E1 = particleenergy(distanceVector)

    # Displace the particle
    dr = SVector{3, Float64}(delta*(rand(rng, Float64) - 0.5), 
                             delta*(rand(rng, Float64) - 0.5), 
                             delta*(rand(rng, Float64) - 0.5))
    
    conf[pointIndex] += dr

    # Update the distance vector and calculate energy
    updatedistance!(conf, box, distanceVector, pointIndex)
    E2 = particleenergy(distanceVector)

    # Get energy difference
    ΔE = E2 - E1
    # Acceptance counter
    accepted = 0

    if rand(rng, Float64) < exp(-ΔE*beta)
        accepted += 1
        E += ΔE
        distanceMatrix[pointIndex, :] = distanceVector
        distanceMatrix[:, pointIndex] = distanceVector
    else
        conf[pointIndex] -= dr
    end
    return(conf, E, accepted, distanceMatrix)
end

"""
mcrun(inputData)

Runs Monte Carlo simulation for a given number of steps
"""
function mcrun(inputData, workerid)
    idString = lpad(workerid + 1, 3, '0')
    energyFile = "energies-p$(idString).dat"
    trajFile = "mctraj-p$(idString).xyz"
    rdfFile = "rdf-p$(idString).dat"

    # Initialize input data
    latticePoints, latticeScaling, σ, steps, Eqsteps, xyzout, outfreq, outlevel, delta, beta, Nbins, binWidth = prepinput(inputData)

    # Generate LJ lattice
    conf, box = ljlattice(latticePoints, latticeScaling)

    # Collect RDF parameters
    rdfParameters = [length(conf), σ, box, binWidth]

    # Initialize the distance histogram
    maxR = Nbins * binWidth
    hist = [LinRange(0, maxR, Nbins), zeros(Int32, Nbins)]

    # Initialize RNG
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Build distance matrix
    distanceMatrix = builddistanceMatrix(conf, box)

    # Initialize the total energy
    E = totalenergy(distanceMatrix)
    #@printf("Starting energy = %.3f epsilon\n\n", E)

    # Save initial configuration and energy
    if outlevel >= 2
        writeenergies(E, 0, false, energyFile)
    end

    # Start writing MC trajectory
    if outlevel == 3
        writexyz(conf, 0, σ, false, trajFile)
    end

    # Acceptance counter
    acceptedTotal = 0

    # Run MC simulation
    @fastmath for i in 1:steps
        conf, E, accepted, distanceMatrix = mcmove!(conf, box, distanceMatrix, E, beta, delta, rng_xor)
        acceptedTotal += accepted

            # MC output
            if i % outfreq == 0 && i > Eqsteps && outlevel >= 1
                hist = hist!(distanceMatrix, hist, binWidth)
            end

            if i % outfreq == 0 && outlevel >= 2
                writeenergies(E, i, true, energyFile)
            end

            if i % xyzout == 0 && outlevel == 3
                writexyz(conf, i, σ, true, trajFile)
            end
    end
    
    acceptanceRatio = acceptedTotal / steps

    if outlevel >= 1
        # Normalize the histogram to the number of frames
        Nframes = (steps - Eqsteps) / outfreq
        hist[2] /= Nframes
        # Write the worker RDF
        writeRDF(rdfFile, hist, rdfParameters)
    end
    
    return(hist, rdfParameters, acceptanceRatio)
end

"""
writexyz(conf, currentStep, σ, append=false, outname, atomtype="Ar")

Writes configuration to an XYZ file
"""
function writexyz(conf, currentStep, σ, append, outname, atomtype="Ar")
    if append
        io = open(outname, "a")
    else
        io = open(outname, "w")
    end
    print(io, length(conf), "\n")
    print(io, "Step = ", @sprintf("%d", currentStep), "\n")
    for i in 1:length(conf)
        print(io, atomtype, " ")
        for j in 1:3
            print(io, @sprintf("%10.3f", conf[i][j]*σ), " ")
            if j == 3
                print(io, "\n")
            end
        end
    end
    close(io)
end

"""
writeenergies(energy, currentStep, append=false, outname)

Writes total energy to an output file
"""
function writeenergies(energy, currentStep, append, outname)
    if append
        io = open(outname, "a")
    else
        io = open(outname, "w")
    end
    print(io, "# Total energy in reduced units \n")
    print(io, "# Step = ", @sprintf("%d", currentStep), "\n")
    print(io, @sprintf("%10.3f", energy), "\n")
    print(io, "\n")
    close(io)
end

"""
writeRDF(outname, hist, rdfParameters)

Normalizes the RDF histogram to RDF and writes into a file
"""
function writeRDF(outname, hist, rdfParameters)
    # Initialize RDF parameters
    N = rdfParameters[1]
    σ = rdfParameters[2]
    box = rdfParameters[3]
    binWidth = rdfParameters[4]

    # Normalize the historgram
    V = (box[1])^3
    Npairs::Int = N*(N-1)/2
    rdfNorm = [(V/Npairs) * 1/(4*π*binWidth*hist[1][i]^2) for i in 2:length(hist[1])]
    RDF = hist[2][2:end] .* rdfNorm
    hist[1] *= σ
    # Write the data
    io = open(outname, "w")
    print(io, "# RDF data \n")
    print(io, "# r, Å; g(r); Histogram \n")
    print(io, @sprintf("%6.3f %12.3f %12.3f", hist[1][1], 0, hist[2][1]), "\n")
    for i in 2:length(hist[1])
        print(io, @sprintf("%6.3f %12.3f %12.3f", hist[1][i], RDF[i-1], hist[2][i]), "\n")
    end
    close(io)
end

"""
readinput(inputname, numParameters=13)

Reads the simulation parameters from
the input file. Assumes a strict
order of the parameters.
"""
function readinput(inputname, numParameters=14)
    inputData = []
    file = open(inputname, "r")
    lines = readlines(file)
    for line in lines
        if length(line) > 0 && line[1] != '#'
            splittedLine = split(line)
            append!(inputData, parse(Float64, splittedLine[3]))
        end
    end
    close(file)

    @assert length(inputData) == numParameters

    return(inputData)
end

"""
prepinput(inputData)

Prepares the input data for MC simulation
"""
function prepinput(inputData)
    # Constants
    kB::Float64 = 1.38064852E-23 # [J/K]
    amu::Float64 = 1.66605304E-27 # [kg]

    # Parse inputData array
    latticePoints::Int = inputData[1] # number of LJ lattice points
    atommass::Float64 = inputData[2] # [amu]
    σ::Float64 = inputData[3] # [Å]
    ϵ::Float64 = inputData[4]*kB # [J]; original value in ϵ/kB [K]
    T::Float64 = inputData[5] # [K]
    density::Float64 = inputData[6] # [kg/m3]
    delta::Float64 = inputData[7] / σ # max displacement [σ]
    steps::Int = inputData[8] # total number of MC steps
    Eqsteps::Int = inputData[9] # equilibration MC steps
    binWidth::Float64= inputData[10] / σ # [σ]
    Nbins::Int = inputData[11] # number of RDF buildDistanceMatrix
    xyzout::Int = inputData[12] # XYZ output frequency
    outfreq::Int = inputData[13] # RDF and E output frequency
    outlevel::Int = inputData[14] # Output level (0: no output, 1: +RDF, 2: +energies, 3: +trajectories)

    # Other parameters
    densityRm::Float64 = (amu*atommass / (2^(1/6) * σ * 1E-10)^3) # Initial density
    beta::Float64 = ϵ/(T*kB) # Inverse reduced temperature
    latticeScaling::Float64 = (densityRm / density)^(1/3) # Scaling to target density

    return(latticePoints, latticeScaling, σ, steps, Eqsteps, xyzout, outfreq, outlevel, delta, beta, Nbins, binWidth)
end
