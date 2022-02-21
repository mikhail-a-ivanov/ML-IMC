using Printf
using RandomNumbers
using StaticArrays


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
    R2 = 0.
    @inbounds for i in 1:length(p1)
        R2 += pbcdx(p1[i], p2[i], box[i])^2
    end
    R = sqrt(R2)
    return(R)
end

"""
ljlattice(latticePoints, latticeScaling)

Generate a 3D latice of LJ atoms
separated by scaled Rm distance 
and the periodic box vectors in reduced units
"""
function ljlattice(latticePoints, latticeScaling)
    lattice = [convert(SVector{3, Float32}, [i, j, k]) 
        for i in 0:latticePoints-1 for j in 0:latticePoints-1 for k in 0:latticePoints-1]
    lattice = lattice .* (2^(1/6) * latticeScaling)
    # Generate PBC box vectors
    boxSide = (latticePoints) * (2^(1/6) * latticeScaling)
    box = convert(SVector{3, Float32}, [boxSide, boxSide, boxSide])
    return(lattice, box)
end

"""
totalenergy(conf, box)

Compute the total potential energy in reduced units
for a given configuration of LJ atoms
"""
function totalenergy(conf, box)
    E = 0.
    for i in 1:length(conf)
        for j in 1:i-1
            r6 = (1/pbcdistance(conf[i], conf[j], box))^6
            r12 = r6^2
            E += 4 * (r12 - r6)
        end
    end
    return(E)
end

"""
particleenergy(conf, pointIndex)

Computes the potential energy of one particle
from a given configuration
"""
function particleenergy(conf, box, pointIndex)
    confCopy = copy(conf)
    particle = conf[pointIndex]
    E = 0.
    # Remove the particle from the configuration
    deleteat!(confCopy, pointIndex)
    @inbounds @fastmath for i in 1:length(confCopy)
        r6 = (1/pbcdistance(particle, confCopy[i], box))^6
        r12 = r6^2
        E += 4 * (r12 - r6)
    end
    return(E)
end

"""
mcmove(E, Tred, conf, box, delta)

Performs a Metropolis Monte Carlo
displacement move
"""
function mcmove(conf, box, E, Tred, delta, rng)
    # Pick a particle at random and calculate its energy
    pointIndex = rand(rng, Int32(1):Int32(length(conf)))
    E1 = particleenergy(conf, box, pointIndex)

    # Displace a particle and compute the new energy
    dr = SVector{3, Float32}(delta*(rand(rng, Float32) - 0.5), delta*(rand(rng, Float32) - 0.5), delta*(rand(rng, Float32) - 0.5))
    conf[pointIndex] += dr
    
    E2 = particleenergy(conf, box, pointIndex)

    # Get energy difference
    ΔE = E2 - E1

    # Acceptance counter
    accepted = 0

    # Accepts or rejects the move
    if ΔE < 0
        accepted += 1
        E += ΔE
    else
        if rand(rng, Float32) < exp(-ΔE/Tred)
            accepted += 1
            E += ΔE
        else
            conf[pointIndex] -= dr
        end
    end
    return(conf, E, accepted)
end

"""
mcrun(steps, outfreq, conf, box, Tred, delta)

Runs Monte Carlo simulation for a given number of steps
"""
function mcrun(steps, outfreq, conf, box, Tred, delta, rng)
    # Initialize energy and configuration arrays
    energies = zeros(steps + 1)
    E = totalenergy(conf, box)
    energies[1] = E

    confs = [[SVector{3, Float32}(zeros(3)) for i in 1:length(conf)] for j in 1:(floor(Int, steps/outfreq) + 1)]
    confs[1] = conf

    # Acceptance counter
    acceptedTotal = 0

    # Run MC simulation
    @inbounds @fastmath for i in 1:steps
        newconf, newE, accepted = mcmove(conf, box, energies[i], Tred, delta, rng)
        energies[i + 1] = newE
        acceptedTotal += accepted
        if i % outfreq == 0
            confs[Int(i/outfreq) + 1] = newconf
        end
    end
    acceptanceRatio = acceptedTotal / steps
    return(confs, energies[1:outfreq:end], acceptanceRatio)
end

"""
writexyz(conf, outname="conf.xyz", atomtype="Ar")

Writes and XYZ file of a system snapshot
for visualization in VMD
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
    T = 95 # target temperature [K]
    density = 1374 # target density [kg/m3]
    delta = 0.2 # Max displacement [σ]
    lattice_points = 10 # Number of lattice points
    # Other parameters
    density_Rm = (amu*atommass / (2^(1/6) * σ * 1E-10)^3) # Initial density
    Tred = T*kB/ϵ # Temperature in reduced units
    lattice_scaling = (density_Rm / density)^(1/3) # Scaling to target density

    # Generate LJ lattice
    conf, box = ljlattice(lattice_points, lattice_scaling)
    println("Box vectors (Å): ", round.(box * σ, digits=3))
    
    # Compute total energy
    E = totalenergy(conf, box)
    @printf("Starting energy = %.3f epsilon\n", E)

    # Save initial configuration to XYZ
    writexyz(conf, 0, σ, false, "start.xyz")

    # Run MC simulation
    steps = Int(1E5) # MC steps
    outfreq = Int(1E4) # Output frequency

    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    confs, energies, acceptanceRatio = mcrun(steps, outfreq, conf, box, Tred, delta, rng_xor)
    println("Acceptance ratio = ", acceptanceRatio)
    println("Energies = ", round.(energies, digits=3), "\n")

    # Save MC trajectory and energies
    writexyz(confs[1], 0, σ, false, "mc-traj.xyz")
    writeenergies(energies[1], 0, false, "energies.dat")
    
    for i in 2:length(energies)
        writeenergies(energies[i], (i-1)*outfreq, true, "energies.dat")
        writexyz(confs[i], (i-1)*outfreq, σ, true, "mc-traj.xyz")
    end

    # Save final configuration to XYZ
    writexyz(confs[end], steps, σ, false, "end.xyz")
    end
end

"""
Run the main() function
"""
run = main()
