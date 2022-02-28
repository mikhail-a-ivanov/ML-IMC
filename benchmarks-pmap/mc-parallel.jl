# Load packages on every worker
@everywhere include("src/mcLJ.jl")
BLAS.set_num_threads(1)

"""
Main function for running MC simulation
"""
function main()
    @time begin
    println("Running MC simulation...")
    
    inputname = ARGS[1]
    inputData = readinput(inputname, 13)

    inputs = [inputData for worker in workers()]
    pmap(mcrun, inputs)

    #println("Acceptance ratio = ", round(acceptanceRatio, digits=3)q)    
    end
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

