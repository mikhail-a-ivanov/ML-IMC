using MPI
using StaticArrays
using Printf

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
"""
function ljlattice(latticePoints, latticeScaling)
    lattice = [convert(SVector{3, Float32}, [i, j, k]) 
        for i in 0:latticePoints-1 for j in 0:latticePoints-1 for k in 0:latticePoints-1]
    scaling::Float32 = (2^(1/6)) * (latticeScaling)
    lattice = lattice .* scaling
    # Generate PBC box vectors
    boxSide::Float32 = (latticePoints) * (2^(1/6) * latticeScaling)
    box::SVector{3, Float32} = [boxSide, boxSide, boxSide]
    return(lattice, box)
end

"""
buildDistanceMatrix(conf, box)

Build distance matrix for a given
configuration
"""
function builddistanceMatrix(conf, box)
    # Build distance matrix
    distanceMatrix = zeros(Float32, length(conf), length(conf))
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


function main()
    @time begin
    MPI.Init()
    comm = MPI.COMM_WORLD

    myid = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)
    root = 0

    E_MPI = zeros(np)

    conf, box = ljlattice(10, 1)
    distanceMatrix = builddistanceMatrix(conf, box)
    # Initialize the total energy
    E = totalenergy(distanceMatrix)
    @printf("Rank %d out of %d: Starting energy = %.3f epsilon\n\n", myid, np, E*(1+myid))


    MPI.Barrier(comm)
    # Collect info
    if myid != root
        MPI.Send(E, 0, 0, comm)
    else
        for i in 1:np
            E_MPI[i] = MPI.Recv(E, 0, 0, comm)
            println("Rank $(i) energy = $(E_MPI)")
        end
    end
    end
    MPI.Finalize()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end