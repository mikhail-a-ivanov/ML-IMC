# Load packages on every worker
@everywhere include("src/mcLJ.jl")

"""
Main function for running MC simulation
"""
function main()
    @time begin
    println("Running MC simulation...")
    
    inputname = ARGS[1]
    inputData = readinput(inputname, 13)

    # Run MC simulation
    @sync for worker in workers()
            @spawnat worker mcrun(inputData)
        end
    
    #println("Acceptance ratio = ", round(acceptanceRatio, digits=3)q)    
    end
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

