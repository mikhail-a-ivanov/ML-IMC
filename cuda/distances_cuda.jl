using CUDA
using Chemfiles
using BenchmarkTools

include("../src/distances.jl")

function reference(frame)
    distanceMatrix = buildDistanceMatrix(frame)
    return distanceMatrix
end

function squaredDistanceComponent(x1, x2, xsize)
    dx = x2 - x1
    dx += -xsize * convert(Int32, round(dx/xsize))
    return dx^2
end

function computeDistance(r1, r2, box)
    return sqrt.(reduce(+, map(squaredDistanceComponent, r1, r2, box)))
end

function distanceVector(r1, coordinates, box)
    return vec(sqrt.(sum(broadcast(squaredDistanceComponent, r1, coordinates, box), dims=1)))
end

function buildDistanceMatrixCPU1(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

    distanceMatrix = Matrix{Float64}(undef, N, N)
    for i in 1:N
		atom_i = coordinates[:, i]
        for j in 1:N
			atom_j = coordinates[:, j]
            distanceMatrix[i, j] = computeDistance(atom_i, atom_j, box)
        end
    end

    return distanceMatrix
end

function buildDistanceMatrixCPU2(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

    distanceMatrix = Matrix{Float64}(undef, N, N)
    for i in 1:N
        distanceMatrix[i, :] = distanceVector(coordinates[:, i], coordinates, box)
    end

    return distanceMatrix
end    

function buildDistanceMatrixCPU3(frame)
    coordinates = positions(frame)
    N::Int64 = length(frame) # number of atoms
    box::Vector{Float64} = lengths(UnitCell(frame)) # pbc box vector

	return sqrt.(sum(broadcast(squaredDistanceComponent, reshape(repeat(coordinates, N), (3, N, N)), coordinates, box), dims=1))[1, :, :]
end    

function buildDistanceMatrixCUDA1(frame)
	coordinates = Matrix{Float32}(positions(frame))
	N::Int32 = length(frame)
	box = Vector{Float32}(lengths(UnitCell(frame)))
	coordinatesGPU = CuArray(coordinates)
	boxGPU = CuArray(box)
	return sqrt.(sum(broadcast(squaredDistanceComponent, reshape(repeat(coordinates, N), (3, N, N)), coordinates, box), dims=1))[1, :, :]
end

function buildDistanceMatrixCUDA2(coordinates, box, N)
	return sqrt.(sum(broadcast(squaredDistanceComponent, reshape(repeat(coordinates, N), (3, N, N)), coordinates, box), dims=1))[1, :, :]
end

function runTests(benchmark=true)
	traj = Trajectory("../methanol-data/100CH3OH/100CH3OH-CG-200.xtc")
	frame = read_step(traj, 1)

	distanceMatrixRef = reference(frame)

	distanceMatrixCPU1 = buildDistanceMatrixCPU1(frame)
	@assert abs(sum(distanceMatrixRef .- distanceMatrixCPU1)) / length(frame) < 1e-8

	distanceMatrixCPU2 = buildDistanceMatrixCPU2(frame)
	@assert abs(sum(distanceMatrixRef .- distanceMatrixCPU2)) / length(frame) < 1e-8

	distanceMatrixCPU3 = buildDistanceMatrixCPU3(frame)
	@assert abs(sum(distanceMatrixRef .- distanceMatrixCPU3)) / length(frame) < 1e-8

	distanceMatrixCUDA1 = buildDistanceMatrixCUDA1(frame)
	@assert abs(sum(Matrix{Float32}(distanceMatrixRef) .- Matrix{Float32}(distanceMatrixCUDA1))) / length(frame) < 1e-3

	# Preparing data for GPU
	coordinates = Matrix{Float32}(positions(frame))
	box = Vector{Float32}(lengths(UnitCell(frame)))
	N = length(frame)
	# Sending data to GPU
	coordinatesGPU = CuArray(coordinates)
	boxGPU = CuArray(box)

	distanceMatrixCUDA2 = buildDistanceMatrixCUDA2(coordinatesGPU, boxGPU, N)
	@assert abs(sum(Matrix{Float32}(distanceMatrixRef) .- Matrix{Float32}(distanceMatrixCUDA2))) / length(frame) < 1e-3

	r1 = coordinates[:, 1]
	distanceVectorCPU = distanceVector(r1, coordinates, box)

	r1GPU = CuArray(r1)
	distanceVectorGPU = distanceVector(r1GPU, coordinatesGPU, boxGPU)
	@assert abs(sum(Vector{Float32}(distanceVectorCPU) .- Vector{Float32}(distanceVectorGPU))) / length(frame) < 1e-3

	if benchmark
		println("Computing distance matrix with Chemfiles distance function on CPU (Float64):")
		@btime reference($frame)

		println("Computing distance matrix with element-wise distance computation on CPU (Float64):")
		@btime buildDistanceMatrixCPU1($frame)

		println("Computing distance matrix with vector-wise distance computation on CPU (Float64):")
		@btime buildDistanceMatrixCPU2($frame)

		println("Computing distance matrix with full vectorization on CPU (Float64):")
		@btime buildDistanceMatrixCPU3($frame)

		println("Computing distance matrix with CUDA (Float32, transfer to/from CPU):")
		@btime CUDA.@sync buildDistanceMatrixCUDA1($frame)

		println("Computing distance matrix with CUDA (Float32, no transfer to/from CPU):")
		@btime CUDA.@sync buildDistanceMatrixCUDA2($coordinatesGPU, $boxGPU, $N)

		println("Computing a single distance vector on CPU (Float64)")
		@btime distanceVector($r1, $coordinates, $box)

		println("Computing a single distance vector on GPU (Float32, no transfer to/from CPU)")
		@btime CUDA.@sync distanceVector($r1GPU, $coordinatesGPU, $boxGPU)
	end
end

runTests()