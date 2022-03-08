# Load packages on every worker
include("src/mcLJ.jl")
BLAS.set_num_threads(1)

"""
Main function for running MC simulation
"""
function main()
    @time begin
    MPI.Init()

    comm = MPI.COMM_WORLD
    workerid = MPI.Comm_rank(comm)
    np = MPI.Comm_size(comm)
    root = 0

    if workerid == root
        println("Running MC simulation on $(np) rank(s)...\n")
    end

    #println("Running MC simulation on rank $(workerid) out of $(np)...")
    
    inputname = ARGS[1]
    inputData = readinput(inputname, 13)

    # Run MC simulation
    hist, rdfParameters, acceptanceRatio = mcrun(inputData, workerid)
    #println("Acceptance ratio = ", round(acceptanceRatio, digits=3))
    
    MPI.Barrier(comm)
    end
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

