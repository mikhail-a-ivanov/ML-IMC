using CUDA
using Chemfiles
using BenchmarkTools

include("src/distances.jl")

function reference(frame)
    distanceMatrix = buildDistanceMatrix(frame)
    return distanceMatrix
end

function pbcdx(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int, round(dx/xsize))
    return dx
end

function computeDistance(r1::Vector{Float64}, r2::Vector{Float64}, box::Vector{Float64})::Float64
    return sqrt(reduce(+, map(x -> x^2, map(pbcdx, r1, r2, box))))
end

function elementFunctionalLoop(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

    distanceMatrix = Matrix{Float64}(undef, N, N)
    for i in 1:N
        for j in 1:N
            distanceMatrix[i, j] = computeDistance(coordinates[:, i], coordinates[:, j], box)
        end
    end

    return distanceMatrix
end

function computeDistanceComponent(x1, x2, boxX)
    return map(x -> x^2, map(pbcdx, x1, x2, boxX))
end

function getDistanceVector(r1::Vector{Float64}, coordinates, box::Vector{Float64})::Vector{Float64}
    return vec(sqrt.(sum(broadcast(computeDistanceComponent, r1, coordinates, box), dims=1)))
end

function vectorFunctionalLoop(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

    distanceMatrix = Matrix{Float64}(undef, N, N)
    for i in 1:N
        distanceMatrix[i, :] = getDistanceVector(coordinates[:, i], coordinates, box)
    end

    return distanceMatrix
end    

function fullVectorization(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector
    return sqrt.(sum(broadcast(computeDistanceComponent, reshape(repeat(coordinates, N), (3, N, N)), coordinates, box), dims=1)) 
end

function fullVectorizationCUDA(frame)
    coordinates = CuArray(positions(frame))
    #N = length(frame) # number of atoms
    box = CuArray(lengths(UnitCell(frame))) # pbc box vector
    return sqrt.(sum(broadcast(computeDistanceComponent, reshape(repeat(coordinates, 512), (3, 512, 512)), coordinates, box), dims=1))[1, :, :]
end

traj = Trajectory("methanol-data/100CH3OH/100CH3OH-CG-200.xtc")
frame = read_step(traj, 1)

distanceMatrixRef = reference(frame)
println("Computing distance matrix with Chemfiles distance function:")
#@btime reference($frame)

println("Computing distance matrix with element-wise distance computation:")
distanceMatrixElementFuncLoop = elementFunctionalLoop(frame)
#@btime elementFunctionalLoop($frame)

@assert abs(sum(distanceMatrixRef .- distanceMatrixElementFuncLoop)) / length(frame) < 1e-8

println("Computing distance matrix with vector-wise distance computation:")
distanceMatrixVectFuncLoop = vectorFunctionalLoop(frame)
#@btime vectorFunctionalLoop($frame)

@assert abs(sum(distanceMatrixRef .- distanceMatrixVectFuncLoop)) / length(frame) < 1e-8

println("Computing distance matrix with a fully vectorized method:")
distanceMatrixFullyVectorized = fullVectorization(frame)
#@btime fullVectorization($frame)

@assert abs(sum(distanceMatrixRef .- distanceMatrixFullyVectorized)) / length(frame) < 1e-8

println("Computing distance matrix with CUDA:")
distanceMatrixCUDA = fullVectorizationCUDA(frame)

@assert abs(sum(distanceMatrixRef .- distanceMatrixCUDA)) / length(frame) < 1e-8