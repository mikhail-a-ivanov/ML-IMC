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
separated by scaled Rm distance in reduced units
"""
function ljlattice(latticePoints, latticeScaling)
    lattice = [convert(Vector{AbstractFloat}, [i, j, k]) 
        for i in 1:latticePoints for j in 1:latticePoints for k in 1:latticePoints]
    lattice = lattice .* (2^(1/6) * latticeScaling)
    return(lattice)
end

"""
writexyz(conf, outname="conf.xyz", atomtype="Ar"

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
Main function for running MC simulation
"""
function main()
    println("Running the main() function...")
    # Initialize parameters
    σ = 3.345 # Å
    ϵ = 125.7 # ϵ/kB, K
    box = [40., 40., 40.]
    
    # Generate LJ lattice
    conf = ljlattice(10, 1)
    

    # Save configuration to XYZ
    writexyz(conf, σ)
end

"""
Run the main() function
"""
run = main()
