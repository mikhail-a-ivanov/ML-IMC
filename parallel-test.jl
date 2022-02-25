using Distributed

println(ARGS[1])

nCPUs = Int(length(Sys.cpu_info())/2)
addprocs(nCPUs)

@everywhere using RandomNumbers

@everywhere function test()
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    println("Hi, I'm worker ", myid(), ".")
    result = rand(rng_xor)
    println(result)
    return(result)
end

function main()
    println("Running on ", length(workers()), " workers.")

    results = zeros(length(workers()))
    for i in 1:length(workers())
        results[i] = remotecall_fetch(test, i)
    end
    
    println(results)
end

run = main()