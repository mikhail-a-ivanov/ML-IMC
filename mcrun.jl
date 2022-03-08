# Load packages on every worker
using Distributed
using Statistics
@everywhere include("src/mcLJ.jl")
BLAS.set_num_threads(1)

"""
Main function for running MC simulation
"""
function main()
    # Total number of workers
    np = length(workers())

    # Start the timer and read the input file
    startTime = Dates.now()
    println("Running MC simulation on $(np) rank(s)...\n")
    println("Starting at: ", startTime)
    
    inputname = ARGS[1]
    inputData = readinput(inputname)

    # Run MC simulation
    inputs = [inputData for worker in workers()]
    outputs = pmap(mcrun, inputs)

    # Write the final RDF
    if inputData[end] >= 1
        rdfRange = outputs[1][1][1]
        rdfParameters = outputs[1][2]
        meanHist = [rdfRange, mean([output[1][2] for output in outputs])]
        writeRDF("rdf-mean.dat", meanHist, rdfParameters)
    end
    
    # Report the mean acceptance ratio
    meanAcceptanceRatio = mean([output[3] for output in outputs])
    println("Mean acceptance ratio = ", round(meanAcceptanceRatio, digits=3))
    
    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    println("Walltime: ", wallTime)
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

