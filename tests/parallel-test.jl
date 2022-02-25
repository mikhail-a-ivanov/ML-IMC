using Distributed
using SharedArrays

nCPUs = Int(length(Sys.cpu_info())/2)
addprocs(nCPUs)

@everywhere using RandomNumbers

@everywhere function test(results)
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    println("Hi, I'm worker ", myid(), ".")
    result = rand(rng_xor)
    println(result)
    results[myid()] = result
end

function main()
    println("Running on ", length(workers()), " workers.")
    results = SharedArray(zeros(6))
    @sync for i in 1:length(workers())
        @spawnat i test(results)
    end

    println(results)
end

run = main()