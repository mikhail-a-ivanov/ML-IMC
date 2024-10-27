using Distributed
@everywhere include("ML_IMC.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    ML_IMC.main()
end
