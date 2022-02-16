using Printf

"""
pbcdx(x1, x2, xsize)

Compute periodic boundary distance between x1 and x2
"""
function pbcdx(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int, round(dx/xsize))
    return(dx)
end

"""
pbcdistance(p1, p2, box)

Compute 3D periodic boundary distance between points p1 and p2 
"""
function pbcdistance(p1, p2, box)
    R2 = 0.
    for i in 1:length(p1)
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
    lattice = [convert(Vector{AbstractFloat}, [i, j, k]) 
        for i in 0:latticePoints-1 for j in 0:latticePoints-1 for k in 0:latticePoints-1]
    lattice = lattice .* (2^(1/6) * latticeScaling)
    # Generate PBC box vectors
    boxSide = (latticePoints) * (2^(1/6) * latticeScaling)
    box = [boxSide, boxSide, boxSide]
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
    for i in 1:length(confCopy)
        r6 = (1/pbcdistance(particle, confCopy[i], box))^6
        r12 = r6^2
        E += 4 * (r12 - r6)
    end
    return(E)
end

"""
writexyz(conf, outname="conf.xyz", atomtype="Ar")

Writes and XYZ file of a system snapshot
for visualization in VMD
"""
function writexyz(conf, σ, outname="conf.xyz", atomtype="Ar")
    io = open(outname, "w")
    print(io, length(conf), "\n\n")
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
mcmove(E, Tred, conf, box, delta)

Performs a Metropolis Monte Carlo
displacement move
"""
function mcmove(conf, box, E, Tred, delta)
    # Pick a particle at random and calculate its energy
    pointIndex = rand(1:length(conf))
    E1 = particleenergy(conf, box, pointIndex)

    # Displace a particle and compute the new energy
    dr = [delta*(rand() - 0.5), delta*(rand() - 0.5), delta*(rand() - 0.5)]
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
        if rand() < exp(-ΔE/Tred)
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
function mcrun(steps, outfreq, conf, box, Tred, delta)
    # Initialize energy and configuration arrays
    energies = zeros(steps + 1)
    E = totalenergy(conf, box)
    energies[1] = E

    confs = [[zeros(3) for i in 1:length(conf)] for j in 1:floor(Int, steps/outfreq)]
    confs[1] = conf

    # Acceptance counter
    acceptedTotal = 0

    # Run MC simulation
    for i in 1:steps
        newconf, newE, accepted = mcmove(conf, box, energies[i], Tred, delta)
        energies[i + 1] = newE
        acceptedTotal += accepted
        if i % outfreq == 0
            confs[Int(i/outfreq)] = newconf
        end
    end
    acceptanceRatio = acceptedTotal / steps
    return(confs, energies[1:outfreq:end], acceptanceRatio)
end
"""
Main function for running MC simulation
"""
function main()
    @time begin
    println("Running MC simulation...")
    # Initialize parameters
    kB = 1.38064852E-23 # [J/K]
    σ = 3.345 # [Å]
    ϵ = 125.7*kB # [J]; original value in ϵ/kB [K]
    T = 95 # [K]
    Tred = T*kB/ϵ # In reduced units
    delta = 0.2 # Max displacement [σ]
    lattice_points = 10
    lattice_scaling = 1

    # Generate LJ lattice
    conf, box = ljlattice(lattice_points, lattice_scaling)
    println("Box vectors (Å): ", round.(box * σ, digits=3))
    
    # Compute total energy
    E = totalenergy(conf, box)
    @printf("Starting energy = %.3f epsilon\n", E)

    # Save configuration to XYZ
    writexyz(conf, σ, "start.xyz")

    # Run MC simulation
    steps = Int(1E6) # MC steps
    outfreq = Int(1E5) # Output frequency
    
    confs, energies, acceptanceRatio = mcrun(steps, outfreq, conf, box, Tred, delta)
    println("Acceptance ratio = ", acceptanceRatio)
    println("Energies = ", round.(energies, digits=3), "\n")

    # Save final configuration to XYZ
    writexyz(confs[end], σ, "end.xyz")
    end
end

"""
Run the main() function
"""
run = main()
