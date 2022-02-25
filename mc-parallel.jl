using Distributed

# Get the number of physical cores
nCPUs = Int(length(Sys.cpu_info())/2)
# Add workers
addprocs(nCPUs)

# load packages on every worker
@everywhere using Printf
@everywhere using RandomNumbers
@everywhere using StaticArrays

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
    R2::Float32 = 0.
    @fastmath @inbounds for i in 1:length(p1)
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
and the distance matrix
"""
function ljlattice(latticePoints, latticeScaling)
    lattice = [convert(SVector{3, Float32}, [i, j, k]) 
        for i in 0:latticePoints-1 for j in 0:latticePoints-1 for k in 0:latticePoints-1]
    scaling::Float32 = (2^(1/6)) * (latticeScaling)
    lattice = lattice .* scaling
    # Generate PBC box vectors
    boxSide::Float32 = (latticePoints) * (2^(1/6) * latticeScaling)
    box::SVector{3, Float32} = [boxSide, boxSide, boxSide]
    # Build distance matrix
    distanceMatrix = zeros(Float32, length(lattice), length(lattice))
    @inbounds for i in 1:length(lattice)
        for j in 1:length(lattice)
            distanceMatrix[i,j] = pbcdistance(lattice[i], lattice[j], box)
        end
    end
    return(lattice, box, distanceMatrix)
end

"""
totalenergy(distanceMatrix)

Compute the total potential energy in reduced units
for a given distance matrix
"""
function totalenergy(distanceMatrix)
    N = convert(Int32, sqrt(length(distanceMatrix)))
    E = 0.
    for i in 1:N
        for j in 1:i-1
            r6 = (1/distanceMatrix[i,j])^6
            r12 = r6^2
            E += 4 * (r12 - r6)
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
    E = 0.
    distanceVector = filter(!iszero, distanceVector)
    @fastmath @inbounds for i in 1:length(distanceVector)
        r6 = (1/distanceVector[i])^6
        r12 = r6^2
        E += 4 * (r12 - r6)
    end
    return(E)
end

"""
hist!(distanceMatrix, hist, binWidth)

Computes RDF histogram
"""
function hist!(distanceMatrix, hist, binWidth)
    N = convert(Int32, sqrt(length(distanceMatrix)))
    for i in 1:N
        for j in 1:i-1
            histIndex = floor(Int32, distanceMatrix[i,j]/binWidth)
            if histIndex <= length(hist[1])
                hist[2][histIndex] += 1
            end
        end
    end
    return(hist)
end

"""
updatedistance(conf, box, distanceVector, pointIndex)

Updates distance vector
"""

function updatedistance(conf, box, distanceVector, pointIndex)
    for i in 1:length(distanceVector)
        distanceVector[i] = pbcdistance(conf[i], conf[pointIndex], box)
    end
    return(distanceVector)
end


"""
mcmove!(conf, box, distanceMatrix, E, Tred, delta, rng)

Performs a Metropolis Monte Carlo
displacement move
"""
function mcmove!(conf, box, distanceMatrix, E, Tred, delta, rng)
    # Pick a particle at random and calculate its energy
    pointIndex = rand(rng, Int32(1):Int32(length(conf)))
    distanceVector = distanceMatrix[:, pointIndex]
    E1 = particleenergy(distanceVector)

    # Displace the particle
    dr = SVector{3, Float32}(delta*(rand(rng, Float32) - 0.5), 
                             delta*(rand(rng, Float32) - 0.5), 
                             delta*(rand(rng, Float32) - 0.5))
    
    conf[pointIndex] += dr

    # Update the distance vector and calculate energy
    newDistanceVector = updatedistance(conf, box, distanceVector, pointIndex)
    E2 = particleenergy(newDistanceVector)

    # Get energy difference
    ΔE = E2 - E1

    # Acceptance counter
    accepted = 0

    # Accepts or rejects the move
    if ΔE < 0
        accepted += 1
        E += ΔE
        # Update the distance matrix 
        # Update both row and column to keep the matrix symmetric
        distanceMatrix[pointIndex, :] = newDistanceVector
        distanceMatrix[:, pointIndex] = newDistanceVector
    else
        if rand(rng, Float32) < exp(-ΔE/Tred)
            accepted += 1
            E += ΔE
            distanceMatrix[pointIndex, :] = newDistanceVector
            distanceMatrix[:, pointIndex] = newDistanceVector
        else
            conf[pointIndex] -= dr
        end
    end
    return(conf, E, accepted, distanceMatrix)
end

"""
mcrun(steps, Eqsteps, outfreq, conf, box, distanceMatrix, Tred, delta, σ, rng, binWidth, Nbins)

Runs Monte Carlo simulation for a given number of steps
"""
function mcrun(steps, Eqsteps, outfreq, conf, box, distanceMatrix, Tred, delta, σ, rng, binWidth, Nbins)
    # Initialize the total energy
    E = totalenergy(distanceMatrix)
    @printf("Starting energy = %.3f epsilon\n\n", E)

    # Save initial configuration and energy
    writexyz(conf, 0, σ, false, "start.xyz")
    writeenergies(E, 0, false, "energies.dat")

    # Start writing MC trajectory
    writexyz(conf, 0, σ, false, "mc-traj.xyz")

    # Acceptance counter
    acceptedTotal = 0

    # Initialize the distance histogram
    maxR = Nbins * binWidth
    hist = [LinRange(0, maxR, Nbins), zeros(Int32, Nbins)]

    # Run MC simulation
    @inbounds @fastmath for i in 1:steps
        conf, E, accepted, distanceMatrix = mcmove!(conf, box, distanceMatrix, E, Tred, delta, rng)
        acceptedTotal += accepted
        if i % outfreq == 0
            writexyz(conf, i, σ, true, "mc-traj.xyz")
            writeenergies(E, i, true, "energies.dat")
            if i > Eqsteps
                hist = hist!(distanceMatrix, hist, binWidth)
            end
            if i % (outfreq*10) == 0
                println("Step ", i, "...")
            end
        end
        if i == steps-1
            writexyz(conf, steps, σ, false, "end.xyz")
        end
    end
    acceptanceRatio = acceptedTotal / steps

    # Normalize the histogram to the number of frames
    Nframes = (steps - Eqsteps) / outfreq
    hist[2] /= Nframes
    return(hist, acceptanceRatio)
end

"""
writexyz(conf, currentStep, σ, append=false, outname="conf.xyz", atomtype="Ar")

Writes configuration to an XYZ file
"""
function writexyz(conf, currentStep, σ, append=false, outname="conf.xyz", atomtype="Ar")
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
writeenergies(energy, currentStep, append=false, outname="energies.dat")

Writes total energy to an output file
"""
function writeenergies(energy, currentStep, append=false, outname="energies.dat")
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
writeRDF(outname, hist, binWidth, σ, N, box)

Normalizes the RDF histogram to RDF and writes into a file
"""
function writeRDF(outname, hist, binWidth, σ, N, box)
    # Normalize the historgram
    V = (box[1])^3
    Npairs = Int32(N*(N-1)/2)
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
Main function for running MC simulation
"""
function main()
    @time begin
    println("Running MC simulation...")
    # Initialize parameters
    # Constants
    kB = 1.38064852E-23 # [J/K]
    amu = 1.66605304E-27 # [kg]
    # Forcefield
    atommass = 39.792 # [amu]
    σ = 3.405 # [Å]
    ϵ = 119.8*kB # [J]; original value in ϵ/kB [K]
    # Conditions
    T = 95.0 # target temperature [K]
    density = 1374.0 # target density [kg/m3]
    delta = 0.5 / σ # Max displacement [Å]
    latticePoints = 10 # Number of lattice points
    steps = Int(1E5) # MC steps
    Eqsteps = Int(1E4) # MC steps
    outfreq = Int(1E4) # Output frequency
    # RDF parameters
    binWidth = 0.1 / σ # in Å
    Nbins = 150

    println("Total number of steps = ", steps)
    println("Equilibration steps = ", Eqsteps)
    println("Output frequency = ", outfreq)

    # Other parameters
    density_Rm = (amu*atommass / (2^(1/6) * σ * 1E-10)^3) # Initial density
    Tred = T*kB/ϵ # Temperature in reduced units
    latticeScaling = (density_Rm / density)^(1/3) # Scaling to target density

    # Generate LJ lattice
    conf, box, distanceMatrix = ljlattice(latticePoints, latticeScaling)
    println("Box vectors (Å): ", round.(box * σ, digits=3))
    
    # Run MC simulation
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    hist, acceptanceRatio = mcrun(steps, Eqsteps, outfreq, conf, box, distanceMatrix, Tred, delta, σ, rng_xor, binWidth, Nbins)
    println("Acceptance ratio = ", round(acceptanceRatio, digits=3))

    # Write RDF into a file
    writeRDF("rdf.dat", hist, binWidth, σ, length(conf), box)
    end
end

"""
Run the main() function
"""
#run = main()
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
