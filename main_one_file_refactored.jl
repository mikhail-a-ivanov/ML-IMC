using Distributed

@everywhere module MlImc
using BSON: @load, @save
using Chemfiles
using Dates
using Distributed
using Flux
using LinearAlgebra
using Printf
using RandomNumbers
using RandomNumbers.Xorshifts
using Statistics

BLAS.set_num_threads(1)

# -----------------------------------------------------------------------------
# --- Structs

struct G2
    eta::Float64
    rcutoff::Float64
    rshift::Float64
end

struct G3
    eta::Float64
    lambda::Float64
    zeta::Float64
    rcutoff::Float64
    rshift::Float64
end

struct G9
    eta::Float64
    lambda::Float64
    zeta::Float64
    rcutoff::Float64
    rshift::Float64
end

struct NeuralNetParameters
    G2Functions::Vector{G2}
    G3Functions::Vector{G3}
    G9Functions::Vector{G9}
    maxDistanceCutoff::Float64
    symmFunctionScaling::Float64
    neurons::Vector{Int}
    iters::Int
    bias::Bool
    activations::Vector{String}
    REGP::Float64
    optimizer::String
    rate::Float64
    momentum::Float64
    decay1::Float64
    decay2::Float64
end

struct SystemParameters
    systemName::String
    trajfile::String
    topname::String
    N::Int
    atomname::String
    rdfname::String
    Nbins::Int
    binWidth::Float64
    T::Float64
    β::Float64
    Δ::Float64
    targetAR::Float64
end

struct MonteCarloParameters
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    trajout::Int
    outfreq::Int
end

struct GlobalParameters
    systemFiles::Vector{String}
    symmetryFunctionFile::String
    mode::String
    outputMode::String
    modelFile::String
    gradientsFile::String
    optimizerFile::String
    adaptiveScaling::Bool
end

struct PreTrainingParameters
    PTsteps::Int64
    PToutfreq::Int64
    PTREGP::Float64
    PToptimizer::String
    PTrate::Float64
    PTmomentum::Float64
    PTdecay1::Float64
    PTdecay2::Float64
end

struct MonteCarloSampleInput
    globalParms::GlobalParameters
    MCParms::MonteCarloParameters
    NNParms::NeuralNetParameters
    systemParms::SystemParameters
    model::Chain
end

struct PreComputedInput
    NNParms::NeuralNetParameters
    systemParms::SystemParameters
    refRDF::Vector{Float64}
end

struct MonteCarloAverages
    descriptor::Vector{Float64}
    energies::Vector{Float64}
    crossAccumulators::Union{Nothing, Vector{Matrix{Float64}}}
    symmFuncMatrixAccumulator::Union{Nothing, Matrix{Float64}}
    acceptanceRatio::Float64
    systemParms::SystemParameters
    mutatedStepAdjust::Float64
end

struct ReferenceData
    distanceMatrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    PMF::Vector{Float64}
    G2Matrices::Vector{Matrix{Float64}}
    G3Matrices::Vector{Matrix{Float64}}
    G9Matrices::Vector{Matrix{Float64}}
end

# -----------------------------------------------------------------------------
# --- Helper Functions (Utils)

function report_optimizer(opt)
    println("Optimizer type: $(typeof(opt))")
    println("   Parameters:")

    param_descriptions = Dict(:eta => "Learning rate",
                              :beta => "Decays",
                              :velocity => "Velocity",
                              :rho => "Momentum coefficient")

    for name in fieldnames(typeof(opt))
        if name ∉ (:state, :velocity)
            value = getfield(opt, name)
            description = get(param_descriptions, name, string(name))
            println("       $description: $value")
        end
    end
end

function print_symmetry_function_info(nn_params)
    for (func_type, functions) in [("G2", nn_params.G2Functions),
        ("G3", nn_params.G3Functions),
        ("G9", nn_params.G9Functions)]
        if !isempty(functions)
            println("    $func_type symmetry functions:")
            println("    eta, Å^-2; rcutoff, Å; rshift, Å")
            for func in functions
                println("       ", func)
            end
        end
    end

    println("Maximum cutoff distance: $(nn_params.maxDistanceCutoff) Å")
    println("Symmetry function scaling parameter: $(nn_params.symmFunctionScaling)")
end

function print_model_summary(model::Chain, nn_params::NeuralNetParameters)
    println(model)
    println("   Number of layers: $(length(nn_params.neurons))")
    println("   Number of neurons in each layer: $(nn_params.neurons)")

    parameter_count = sum(sum(length, Flux.params(layer)) for layer in model)
    println("   Total number of parameters: $parameter_count")
    println("   Using bias parameters: $(nn_params.bias)")
end

function check_file(filename::AbstractString)
    isfile(filename) || throw(ArgumentError("Could not locate file: $filename"))
end

function read_xtc(system_params::SystemParameters)
    check_file(system_params.trajfile)
    return Trajectory(system_params.trajfile)
end

function read_pdb(system_params::SystemParameters)
    check_file(system_params.topname)
    return Trajectory(system_params.topname)
end

function write_rdf(outname::AbstractString, rdf::Vector{Float64}, system_params::SystemParameters)
    bins = [bin * system_params.binWidth for bin in 1:(system_params.Nbins)]

    open(outname, "w") do io
        println(io, "# System: $(system_params.systemName)")
        println(io, "# RDF data ($(system_params.atomname) - $(system_params.atomname))")
        println(io, "# r, Å; g(r);")
        for (bin, g) in zip(bins, rdf)
            println(io, @sprintf("%6.4f %12.4f", bin, g))
        end
    end

    check_file(outname)
end

function write_energies(outname::AbstractString, energies::Vector{Float64},
                        mc_params::MonteCarloParameters, system_params::SystemParameters,
                        slicing::Int=1)
    steps = 0:(mc_params.outfreq * slicing):(mc_params.steps)
    sliced_energies = energies[1:slicing:end]

    open(outname, "w") do io
        println(io, "# System: $(system_params.systemName)")
        println(io, @sprintf("# %8s %22s", "Step", "Total energy, kJ/mol"))
        for (step, energy) in zip(steps, sliced_energies)
            println(io, @sprintf("%9d %10.4f", step, energy))
        end
    end

    check_file(outname)
end

function write_trajectory(conf::Matrix{Float64}, box::Vector{Float64},
                          system_params::SystemParameters, outname::AbstractString,
                          mode::Char='w')
    frame = Frame()
    box_center = box ./ 2
    set_cell!(frame, UnitCell(box))

    for i in 1:(system_params.N)
        wrapped_atom_coords = wrap!(UnitCell(frame), view(conf, :, i)) .+ box_center
        add_atom!(frame, Atom(system_params.atomname), wrapped_atom_coords)
    end

    Trajectory(outname, string(mode)) do traj
        write(traj, frame)
    end

    check_file(outname)
end

function read_rdf(rdfname::AbstractString)
    check_file(rdfname)

    bins = Float64[]
    rdf = Float64[]

    open(rdfname, "r") do file
        for line in eachline(file)
            stripped_line = strip(line)
            isempty(stripped_line) && continue  # Skip empty lines
            startswith(stripped_line, "#") && continue  # Skip comment lines

            values = split(stripped_line)
            length(values) < 2 && continue  # Skip lines with insufficient data

            push!(bins, parse(Float64, values[1]))
            push!(rdf, parse(Float64, values[2]))
        end
    end

    isempty(bins) && @warn "No data found in $rdfname"

    return bins, rdf
end

function build_distance_matrix_chemfiles(frame::Frame)::Matrix{Float64}
    n_atoms = length(frame)
    distance_matrix = Matrix{Float64}(undef, n_atoms, n_atoms)

    @inbounds for i in 1:n_atoms
        distance_matrix[i, i] = 0.0
        for j in (i + 1):n_atoms
            dist = distance(frame, i - 1, j - 1)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        end
    end

    return distance_matrix
end

function update_distance!(frame::Frame, distance_vector::Vector{Float64}, point_index::Int)::Vector{Float64}
    @inbounds for i in eachindex(distance_vector)
        distance_vector[i] = distance(frame, i - 1, point_index - 1) # NOTE: Chemfiles.distance 0-based index
    end
    return distance_vector
end

function distance_cutoff(distance::Float64, r_cutoff::Float64=6.0)::Float64
    if distance > r_cutoff
        return 0.0
    else
        return (0.5 * (cos(π * distance / r_cutoff) + 1.0))
    end
end

function compute_distance_component(x1::Float64, x2::Float64, box_size::Float64)::Float64
    dx = x2 - x1
    dx -= box_size * round(dx / box_size)
    return dx
end

function compute_squared_distance_component(x1::Float64, x2::Float64, box_size::Float64)::Float64
    dx = x2 - x1
    dx -= box_size * round(dx / box_size)
    return dx * dx
end

function compute_directional_vector(r1::AbstractVector{T}, r2::AbstractVector{T},
                                    box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    return [compute_distance_component(r1[i], r2[i], box[i]) for i in eachindex(r1, r2, box)]
end

function compute_distance(r1::AbstractVector{T}, r2::AbstractVector{T},
                          box::AbstractVector{T})::T where {T <: AbstractFloat}
    return sqrt(sum(compute_squared_distance_component(r1[i], r2[i], box[i]) for i in eachindex(r1, r2, box)))
end

function compute_distance_vector(r1::AbstractVector{T}, coordinates::AbstractMatrix{T},
                                 box::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    return [sqrt(sum(compute_squared_distance_component(r1[i], coordinates[i, j], box[i]) for i in axes(coordinates, 1)))
            for j in axes(coordinates, 2)]
end

function build_distance_matrix(frame::Frame)::Matrix{Float64}
    coordinates = positions(frame)
    n_atoms = length(frame)
    box = lengths(UnitCell(frame))

    distance_matrix = Matrix{Float64}(undef, n_atoms, n_atoms)

    @inbounds for i in 1:n_atoms
        distance_matrix[i, i] = 0.0
        for j in (i + 1):n_atoms
            dist = compute_distance(coordinates[:, i], coordinates[:, j], box)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        end
    end

    return distance_matrix
end

# -----------------------------------------------------------------------------
# --- Initialization

function parameters_init()
    # NOTE: Not refactored!!!
    # Read the input name from the command line argument
    if length(ARGS) > 0
        inputname = ARGS[1]
        # Check if no input file is provided
        # or if ML-IMC is started from jupyter
        # in the latter case the first argument has "json" extension
    end
    if length(ARGS) == 0 || occursin("json", ARGS[1])
        println("No input file was provided!")
        println("Trying to read input data from ML-IMC-init.in")
        inputname = "ML-IMC-init.in"
    end

    # Constants
    NA::Float64 = 6.02214076 # [mol-1] * 10^-23
    kB::Float64 = 1.38064852 * NA / 1000 # [kJ/(mol*K)]

    # Read the input file
    check_file(inputname)
    file = open(inputname, "r")
    lines = readlines(file)
    splittedLines = [split(line) for line in lines]

    # Make a list of field names
    globalFields = [String(field) for field in fieldnames(GlobalParameters)]
    MCFields = [String(field) for field in fieldnames(MonteCarloParameters)]
    NNFields = [String(field) for field in fieldnames(NeuralNetParameters)]
    preTrainFields = [String(field) for field in fieldnames(PreTrainingParameters)]

    # Input variable arrays
    globalVars = []
    MCVars = []
    NNVars = []
    preTrainVars = []
    G2s = []
    G3s = []
    G9s = []

    # Loop over fieldnames and fieldtypes and over splitted lines
    # Global parameters
    for (field, fieldtype) in zip(globalFields, fieldtypes(GlobalParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if field == "systemFiles"
                    systemFiles = []
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(systemFiles, [strip(element, ',')])
                        elseif element == "#"
                            break
                        end
                    end
                    append!(globalVars, [systemFiles])
                else
                    if fieldtype != String
                        append!(globalVars, parse(fieldtype, line[3]))
                    else
                        append!(globalVars, [line[3]])
                    end
                end
            end
        end
    end
    globalParms = GlobalParameters(globalVars...)

    # Symmetry functions
    maxDistanceCutoff = 0.0
    scalingFactor = 1.0
    check_file(globalParms.symmetryFunctionFile)
    symmetryFunctionFile = open(globalParms.symmetryFunctionFile, "r")
    symmetryFunctionLines = readlines(symmetryFunctionFile)
    splittedSymmetryFunctionLines = [split(line) for line in symmetryFunctionLines]
    for line in splittedSymmetryFunctionLines
        if length(line) != 0 && line[1] == "G2"
            G2Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G2))
                append!(G2Parameters, parse(fieldtype, line[fieldIndex + 1]))
            end
            append!(G2s, [G2(G2Parameters...)])
        end
        if length(line) != 0 && line[1] == "G3"
            G3Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G3))
                append!(G3Parameters, parse(fieldtype, line[fieldIndex + 1]))
            end
            append!(G3s, [G3(G3Parameters...)])
        end
        if length(line) != 0 && line[1] == "G9"
            G9Parameters = []
            for (fieldIndex, fieldtype) in enumerate(fieldtypes(G9))
                append!(G9Parameters, parse(fieldtype, line[fieldIndex + 1]))
            end
            append!(G9s, [G9(G9Parameters...)])
        end
        if length(line) != 0 && line[1] == "scaling"
            scalingFactor = parse(Float64, line[3])
        end
    end
    # Set maximum distance cutoff
    for G2Function in G2s
        if maxDistanceCutoff < G2Function.rcutoff
            maxDistanceCutoff = G2Function.rcutoff
        end
    end
    for G3Function in G3s
        if maxDistanceCutoff < G3Function.rcutoff
            maxDistanceCutoff = G3Function.rcutoff
        end
    end
    for G9Function in G9s
        if maxDistanceCutoff < G9Function.rcutoff
            maxDistanceCutoff = G9Function.rcutoff
        end
    end

    # MC parameters
    for (field, fieldtype) in zip(MCFields, fieldtypes(MonteCarloParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if fieldtype != String
                    append!(MCVars, parse(fieldtype, line[3]))
                else
                    append!(MCVars, [line[3]])
                end
            end
        end
    end
    MCParms = MonteCarloParameters(MCVars...)

    # Loop over fieldnames and fieldtypes and over splitted lines
    for (field, fieldtype) in zip(NNFields, fieldtypes(NeuralNetParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if field == "neurons"
                    inputNeurons = length(G2s) + length(G3s) + length(G9s)
                    neurons = [inputNeurons]
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(neurons, parse(Int, strip(element, ',')))
                        elseif element == "#"
                            break
                        end
                    end
                    append!(NNVars, [neurons])
                elseif field == "activations"
                    activations = []
                    for (elementId, element) in enumerate(line)
                        if elementId > 2 && element != "#"
                            append!(activations, [strip(element, ',')])
                        elseif element == "#"
                            break
                        end
                    end
                    append!(NNVars, [activations])
                else
                    if fieldtype != String
                        append!(NNVars, parse(fieldtype, line[3]))
                    else
                        append!(NNVars, [line[3]])
                    end
                end
            end
        end
    end
    NNParms = NeuralNetParameters(G2s, G3s, G9s, maxDistanceCutoff, scalingFactor, NNVars...)

    # Pre-training parameters
    for (field, fieldtype) in zip(preTrainFields, fieldtypes(PreTrainingParameters))
        for line in splittedLines
            if length(line) != 0 && field == line[1]
                if fieldtype != String
                    append!(preTrainVars, parse(fieldtype, line[3]))
                else
                    append!(preTrainVars, [line[3]])
                end
            end
        end
    end
    preTrainParms = PreTrainingParameters(preTrainVars...)

    # Read system input files
    systemParmsList = [] # list of systemParameters structs
    systemFields = [String(field) for field in fieldnames(SystemParameters)]
    for inputname in globalParms.systemFiles
        systemVars = []
        check_file(inputname)
        file = open(inputname, "r")
        lines = readlines(file)
        splittedLines = [split(line) for line in lines]
        for (field, fieldtype) in zip(systemFields, fieldtypes(SystemParameters))
            for line in splittedLines
                if length(line) != 0 && field == line[1]
                    if field == "T"
                        T = parse(fieldtype, line[3])
                        β = 1 / (kB * T)
                        append!(systemVars, T)
                        append!(systemVars, β)
                    elseif field == "topname"
                        topname = [line[3]]
                        check_file(topname[1])
                        pdb = Trajectory("$(topname[1])")
                        pdbFrame = read(pdb)
                        N = length(pdbFrame)
                        atomname = name(Atom(pdbFrame, 1))
                        append!(systemVars, topname)
                        append!(systemVars, N)
                        append!(systemVars, [atomname])
                    elseif field == "rdfname"
                        rdfname = [line[3]]
                        bins, rdf = read_rdf("$(rdfname[1])")
                        Nbins = length(bins)
                        binWidth = bins[1]
                        append!(systemVars, [rdfname[1]])
                        append!(systemVars, Nbins)
                        append!(systemVars, binWidth)
                    else
                        if fieldtype != String
                            append!(systemVars, parse(fieldtype, line[3]))
                        else
                            append!(systemVars, [line[3]])
                        end
                    end
                end
            end
        end
        systemParms = SystemParameters(systemVars...)
        append!(systemParmsList, [systemParms])
    end

    if globalParms.mode == "training"
        println("Running ML-IMC in the training mode.")
    else
        println("Running ML-IMC in the simulation mode.")
    end

    return (globalParms, MCParms, NNParms, preTrainParms, systemParmsList)
end

function input_init(global_params::GlobalParameters, nn_params::NeuralNetParameters,
                    pretrain_params::PreTrainingParameters, system_params_list)
    # NOTE: Not refactored!!!
    # Read reference data
    refRDFs = []
    for systemParms in system_params_list
        bins, refRDF = read_rdf(systemParms.rdfname)
        append!(refRDFs, [refRDF])
    end

    # Initialize the model and the optimizer
    if global_params.modelFile == "none"
        # Initialize the model
        println("Initializing a new neural network with random weigths")

        model = model_init(nn_params)

        if global_params.optimizerFile != "none"
            println("Ignoring given optimizer filename...")
        end
        if global_params.gradientsFile != "none"
            println("Ignoring given gradients filename...")
        end
        # Run pre-training if no initial model is given
        opt = init_optimizer(pretrain_params)
        # Restart the training
    else
        # Loading the model
        check_file(global_params.modelFile)
        println("Reading model from $(global_params.modelFile)")
        @load global_params.modelFile model
        if global_params.mode == "training"
            # Either initialize the optimizer or read from a file
            if global_params.optimizerFile != "none"
                check_file(global_params.optimizerFile)
                println("Reading optimizer state from $(global_params.optimizerFile)")
                @load global_params.optimizerFile opt
            else
                opt = init_optimizer(nn_params)
            end
            # Optionally read gradients from a file
            if global_params.gradientsFile != "none"
                check_file(global_params.gradientsFile)
                println("Reading gradients from $(global_params.gradientsFile)")
                @load global_params.gradientsFile meanLossGradients
            end
            # Update the model if both opt and gradients are restored
            if global_params.optimizerFile != "none" && global_params.gradientsFile != "none"
                println("\nUsing the restored gradients and optimizer to update the current model...\n")
                update_model!(model, opt, meanLossGradients)

                # Skip updating if no gradients are provided
            elseif global_params.optimizerFile != "none" && global_params.gradientsFile == "none"
                println("\nNo gradients were provided, rerunning the training iteration with the current model and restored optimizer...\n")

                # Update the model if gradients are provided without the optimizer:
                # valid for optimizer that do not save their state, e.g. Descent,
                # otherwise might produce unexpected results
            elseif global_params.optimizerFile == "none" && global_params.gradientsFile != "none"
                println("\nUsing the restored gradients with reinitialized optimizer to update the current model...\n")
                update_model!(model, opt, meanLossGradients)
            else
                println("\nNeither gradients nor optimizer were provided, rerunning the training iteration with the current model...\n")
            end
        end
    end

    if global_params.mode == "training"
        return (model, opt, refRDFs)
    else
        return (model)
    end
end

function init_optimizer(params::Union{NeuralNetParameters, PreTrainingParameters})
    function get_rate(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.rate : params.PTrate
    end

    function get_momentum(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.momentum : params.PTmomentum
    end

    function get_decay1(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.decay1 : params.PTdecay1
    end

    function get_decay2(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.decay2 : params.PTdecay2
    end

    OPTIMIZER_MAP = Dict("Momentum" => Momentum,
                         "Descent" => Descent,
                         "Nesterov" => Nesterov,
                         "RMSProp" => RMSProp,
                         "Adam" => Adam,
                         "RAdam" => RAdam,
                         "AdaMax" => AdaMax,
                         "AdaGrad" => AdaGrad,
                         "AdaDelta" => AdaDelta,
                         "AMSGrad" => AMSGrad,
                         "NAdam" => NAdam,
                         "AdamW" => AdamW,
                         "OAdam" => OAdam,
                         "AdaBelief" => AdaBelief)

    optimizer_name = params isa NeuralNetParameters ? params.optimizer : params.PToptimizer
    optimizer_func = get(OPTIMIZER_MAP, optimizer_name, Descent)

    if optimizer_func in (Momentum, Nesterov, RMSProp)
        return optimizer_func(get_rate(params), get_momentum(params))
    elseif optimizer_func in (Adam, RAdam, AdaMax, AMSGrad, NAdam, AdamW, OAdam, AdaBelief)
        return optimizer_func(get_rate(params), (get_decay1(params), get_decay2(params)))
    else
        return optimizer_func(get_rate(params))
    end
end

# -----------------------------------------------------------------------------
# --- Energy and Gradients

function compute_atomic_energy(input_layer::AbstractVector{T}, model::Flux.Chain)::T where {T <: AbstractFloat}
    return only(model(input_layer))
end

function compute_system_total_energy_scalar(symm_func_matrix::AbstractMatrix{T},
                                            model::Flux.Chain) where {T <: AbstractFloat}
    return sum(compute_atomic_energy(row, model) for row in eachrow(symm_func_matrix))
end

function update_system_energies_vector(symm_func_matrix::AbstractMatrix{T},
                                       model::Flux.Chain,
                                       indices_for_update::AbstractVector{Bool},
                                       previous_energies::AbstractVector{T}) where {T <: AbstractFloat}
    updated_energies = copy(previous_energies)
    update_indices = findall(indices_for_update)

    if !isempty(update_indices)
        new_energies = [compute_atomic_energy(symm_func_matrix[i, :], model) for i in update_indices]
        updated_energies[update_indices] .= new_energies
    end

    return updated_energies
end

function get_energies_update_mask(distance_vector::AbstractVector{T},
                                  nn_params)::Vector{Bool} where {T <: AbstractFloat}
    return distance_vector .< nn_params.maxDistanceCutoff
end

function init_system_energies_vector(symm_func_matrix::AbstractMatrix{T}, model::Flux.Chain) where {T <: AbstractFloat}
    return [compute_atomic_energy(row, model) for row in eachrow(symm_func_matrix)]
end

function compute_energy_gradients(symm_func_matrix::AbstractMatrix{T},
                                  model::Flux.Chain,
                                  nn_params::NeuralNetParameters)::Vector{AbstractArray{T}} where {T <: AbstractFloat}
    energy_gradients = Vector{AbstractArray{T}}()

    gs = gradient(compute_system_total_energy_scalar, symm_func_matrix, model)
    # Structure: gs[2][1][layerId][1 - weigths; 2 - biases]

    for layer_gradients in gs[2][1]
        push!(energy_gradients, layer_gradients[1])  # weights
        if nn_params.bias
            push!(energy_gradients, layer_gradients[2])  # biases
        end
    end

    return energy_gradients
end

function compute_cross_correlation(descriptor::Vector{T},
                                   energy_gradients::Vector{<:AbstractArray{T}})::Vector{Matrix{T}} where {T <:
                                                                                                           AbstractFloat}
    cross_correlations = Vector{Matrix{T}}(undef, length(energy_gradients))
    for (i, gradient) in enumerate(energy_gradients)
        cross_correlations[i] = descriptor * gradient[:]' # Matrix Nbins x Nparameters
    end
    return cross_correlations
end

function initialize_cross_accumulators(nn_params::NeuralNetParameters,
                                       system_params::SystemParameters)::Vector{Matrix{Float64}}
    n_layers = length(nn_params.neurons)
    cross_accumulators = Vector{Matrix{Float64}}()

    for layer_id in 2:n_layers
        weights_shape = (system_params.Nbins, nn_params.neurons[layer_id - 1] * nn_params.neurons[layer_id])
        push!(cross_accumulators, zeros(weights_shape))

        if nn_params.bias
            bias_shape = (system_params.Nbins, nn_params.neurons[layer_id])
            push!(cross_accumulators, zeros(bias_shape))
        end
    end

    return cross_accumulators
end

function update_cross_accumulators!(cross_accumulators::Vector{Matrix{T}},
                                    symm_func_matrix::Matrix{T},
                                    descriptor::Vector{T},
                                    model::Chain,
                                    nn_params::NeuralNetParameters)::Vector{Matrix{T}} where {T <: AbstractFloat}
    energy_gradients = compute_energy_gradients(symm_func_matrix, model, nn_params)
    new_cross_correlations = compute_cross_correlation(descriptor, energy_gradients)

    @inbounds for i in eachindex(cross_accumulators, new_cross_correlations)
        cross_accumulators[i] .+= new_cross_correlations[i]
    end

    return cross_accumulators
end

function compute_ensemble_correlation(symm_func_matrix::Matrix{T},
                                      descriptor::Vector{T},
                                      model::Chain,
                                      nn_params::NeuralNetParameters)::Vector{Matrix{T}} where {T <: AbstractFloat}
    energy_gradients = compute_energy_gradients(symm_func_matrix, model, nn_params)
    ensemble_correlations = compute_cross_correlation(descriptor, energy_gradients)
    return ensemble_correlations
end

function compute_descriptor_gradients(cross_accumulators::Vector{Matrix{T}},
                                      ensemble_correlations::Vector{Matrix{T}},
                                      system_params::SystemParameters)::Vector{Matrix{T}} where {T <: AbstractFloat}
    descriptor_gradients = Vector{Matrix{T}}(undef, length(cross_accumulators))
    for i in eachindex(cross_accumulators, ensemble_correlations)
        descriptor_gradients[i] = -system_params.β .* (cross_accumulators[i] - ensemble_correlations[i])
    end
    return descriptor_gradients
end

function compute_loss_gradients(cross_accumulators::Vector{Matrix{T}},
                                symm_func_matrix::Matrix{T},
                                descriptor_nn::Vector{T},
                                descriptor_ref::Vector{T},
                                model::Chain,
                                system_params::SystemParameters,
                                nn_params::NeuralNetParameters)::Vector{AbstractArray{T}} where {T <: AbstractFloat}
    ensemble_correlations = compute_ensemble_correlation(symm_func_matrix, descriptor_nn, model, nn_params)
    descriptor_gradients = compute_descriptor_gradients(cross_accumulators, ensemble_correlations, system_params)

    dLdS = @. 2 * (descriptor_nn - descriptor_ref)

    loss_gradients = Vector{AbstractArray{T}}(undef, length(Flux.params(model)))

    for (i, (gradient, parameters)) in enumerate(zip(descriptor_gradients, Flux.params(model)))
        loss_gradient = reshape(dLdS' * gradient, size(parameters))
        reg_loss_gradient = @. 2 * nn_params.REGP * parameters
        loss_gradients[i] = loss_gradient .+ reg_loss_gradient
    end

    return loss_gradients
end

function update_model!(model::Chain,
                       optimizer::Flux.Optimise.AbstractOptimiser,
                       loss_gradients::Vector{<:AbstractArray{T}}) where {T <: AbstractFloat}
    for (gradient, parameter) in zip(loss_gradients, Flux.params(model))
        Flux.Optimise.update!(optimizer, parameter, gradient)
    end

    return model
end

# -----------------------------------------------------------------------------
# --- Training

function compute_training_loss(descriptor_nn::AbstractVector{T},
                               descriptor_ref::AbstractVector{T},
                               model::Flux.Chain,
                               nn_params::NeuralNetParameters,
                               mean_max_displacement::T) where {T <: AbstractFloat}

    # Compute descriptor difference loss
    descriptor_loss = sum(abs2, descriptor_nn .- descriptor_ref)

    # Compute L2 regularization loss if regularization parameter is positive
    reg_loss = zero(T)
    if nn_params.REGP > zero(T)
        reg_loss = nn_params.REGP * sum(sum(abs2, p) for p in Flux.params(model))
    end

    total_loss = descriptor_loss + reg_loss

    # Print loss components with consistent formatting
    for (label, value) in [
        ("Regularization Loss", reg_loss),
        ("Descriptor Loss", descriptor_loss),
        ("Total Loss", total_loss),
        ("Max displacement", mean_max_displacement)
    ]
        println("  $label = $(round(value; digits=8))")
    end

    # Log descriptor loss to file
    LOSS_LOG_FILE = "training-loss-values.out"
    try
        open(LOSS_LOG_FILE, "a") do io
            println(io, round(descriptor_loss; digits=8))
        end
        check_file(LOSS_LOG_FILE)
    catch e
        @warn "Failed to log loss value" exception=e
    end

    return total_loss
end

function build_network(nn_params::NeuralNetParameters)
    return [(nn_params.neurons[i - 1], nn_params.neurons[i],
             getfield(Flux, Symbol(nn_params.activations[i - 1])))
            for i in 2:length(nn_params.neurons)]
end

function build_chain(nn_params::NeuralNetParameters, layers...)
    return Chain([nn_params.bias ? Dense(layer...) : Dense(layer..., bias=false)
                  for layer in layers]...)
end

function model_init(nn_params::NeuralNetParameters)
    println("Building a model...")

    network = build_network(nn_params)
    model = build_chain(nn_params, network...)
    model = f64(model)

    print_model_summary(model, nn_params)

    return model
end

function prepare_monte_carlo_inputs(global_params::GlobalParameters,
                                    mc_params::MonteCarloParameters,
                                    nn_params::NeuralNetParameters,
                                    system_params_list,
                                    model::Flux.Chain)
    n_systems = length(system_params_list)
    n_workers = nworkers()

    # Validate that workers can be evenly distributed across systems
    if n_workers % n_systems != 0
        throw(ArgumentError("Number of workers ($n_workers) must be divisible by number of systems ($n_systems)"))
    end

    # Create input for each system
    reference_inputs = Vector{MonteCarloSampleInput}(undef, n_systems)
    for (i, system_params) in enumerate(system_params_list)
        reference_inputs[i] = MonteCarloSampleInput(global_params,
                                                    mc_params,
                                                    nn_params,
                                                    system_params,
                                                    model)
    end

    # Replicate inputs for all workers
    sets_per_system = n_workers ÷ n_systems
    return repeat(reference_inputs, sets_per_system)
end

function collect_system_averages(outputs, ref_rdfs, system_params_list, global_params, nn_params, model)
    meanLoss = 0.0
    systemOutputs = []

    systemLosses = []

    for (systemId, systemParms) in enumerate(system_params_list)
        println("   System $(systemParms.systemName):")
        systemLoss = 0.0

        meanDescriptor = []
        meanEnergies = []
        if global_params.mode == "training"
            meanCrossAccumulators = []
            meansymmFuncMatrixAccumulator = []
        end
        meanAcceptanceRatio = []
        meanMaxDisplacement = []
        # Find the corresponding outputs
        for outputID in eachindex(outputs)
            # Save the outputID if the system names from input and output match
            if systemParms.systemName == outputs[outputID].systemParms.systemName
                append!(meanDescriptor, [outputs[outputID].descriptor])
                append!(meanEnergies, [outputs[outputID].energies])
                if global_params.mode == "training"
                    append!(meanCrossAccumulators, [outputs[outputID].crossAccumulators])
                    append!(meansymmFuncMatrixAccumulator, [outputs[outputID].symmFuncMatrixAccumulator])
                end
                append!(meanAcceptanceRatio, [outputs[outputID].acceptanceRatio])
                append!(meanMaxDisplacement, [outputs[outputID].mutatedStepAdjust])
            end
        end
        # Take averages from each worker
        meanDescriptor = mean(meanDescriptor)
        meanEnergies = mean(meanEnergies)
        if global_params.mode == "training"
            meanCrossAccumulators = mean(meanCrossAccumulators)
            meansymmFuncMatrixAccumulator = mean(meansymmFuncMatrixAccumulator)
        end
        meanAcceptanceRatio = mean(meanAcceptanceRatio)
        meanMaxDisplacement = mean(meanMaxDisplacement)
        if global_params.mode == "training"
            systemOutput = MonteCarloAverages(meanDescriptor, meanEnergies, meanCrossAccumulators,
                                              meansymmFuncMatrixAccumulator, meanAcceptanceRatio, systemParms,
                                              meanMaxDisplacement)
        else
            systemOutput = MonteCarloAverages(meanDescriptor, meanEnergies, nothing, nothing, meanAcceptanceRatio,
                                              systemParms,
                                              meanMaxDisplacement)
        end
        # Compute loss and print some output info
        println("       Acceptance ratio = ", round(meanAcceptanceRatio; digits=4))
        println("       Max displacement = ", round(meanMaxDisplacement; digits=4))
        if global_params.mode == "training"
            systemLoss = compute_training_loss(systemOutput.descriptor, ref_rdfs[systemId], model, nn_params,
                                               meanMaxDisplacement)
            meanLoss += systemLoss
            append!(systemLosses, systemLoss)
        end

        append!(systemOutputs, [systemOutput])
    end
    if global_params.mode == "training"
        meanLoss /= length(system_params_list)
        println("   \nTotal Average Loss = ", round(meanLoss; digits=8))
    end
    return (systemOutputs, systemLosses)
end

function compute_adaptive_gradient_coefficients(system_losses::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    isempty(system_losses) && throw(ArgumentError("System losses vector cannot be empty"))

    max_loss = maximum(system_losses)
    if iszero(max_loss)
        n_systems = length(system_losses)
        return fill(one(T) / n_systems, n_systems)
    end

    # Normalize losses relative to maximum loss
    relative_coefficients = system_losses ./ max_loss

    # Scale coefficients to sum to 1.0
    normalization_factor = one(T) / sum(relative_coefficients)

    return relative_coefficients .* normalization_factor
end

function train!(global_params, mc_params, nn_params, system_params_list, model, optimizer, ref_rdfs)
    # Run training iterations
    iteration = 1

    while iteration <= nn_params.iters
        iterString = lpad(iteration, 2, '0')
        println("\nIteration $(iteration)...")

        # Prepare multi-reference inputs
        inputs = prepare_monte_carlo_inputs(global_params, mc_params, nn_params, system_params_list, model)

        # Run the simulation in parallel
        outputs = pmap(mcsample!, inputs)

        # Collect averages corresponding to each reference system
        systemOutputs, systemLosses = collect_system_averages(outputs, ref_rdfs, system_params_list, global_params,
                                                              nn_params,
                                                              model)

        # Compute loss and the gradients
        lossGradients = []
        for (systemId, systemOutput) in enumerate(systemOutputs)
            systemParms = system_params_list[systemId]
            lossGradient = compute_loss_gradients(systemOutput.crossAccumulators,
                                                  systemOutput.symmFuncMatrixAccumulator,
                                                  systemOutput.descriptor, ref_rdfs[systemId], model, systemParms,
                                                  nn_params)

            append!(lossGradients, [lossGradient])
            # Write descriptors and energies
            name = systemParms.systemName
            write_rdf("RDFNN-$(name)-iter-$(iterString).dat", systemOutput.descriptor, systemParms)
            write_energies("energies-$(name)-iter-$(iterString).dat", systemOutput.energies, mc_params, systemParms, 1)
        end
        # Average the gradients
        if global_params.adaptiveScaling
            gradientCoeffs = compute_adaptive_gradient_coefficients(systemLosses)
            println("\nGradient scaling: \n")
            for (gradientCoeff, systemParms) in zip(gradientCoeffs, system_params_list)
                println("   System $(systemParms.systemName): $(round(gradientCoeff, digits=8))")
            end

            meanLossGradients = sum(lossGradients .* gradientCoeffs)
        else
            meanLossGradients = mean([lossGradient for lossGradient in lossGradients])
        end

        # Write the model and opt (before training!) and the gradients
        @save "model-iter-$(iterString).bson" model
        check_file("model-iter-$(iterString).bson")

        @save "opt-iter-$(iterString).bson" optimizer
        check_file("opt-iter-$(iterString).bson")

        @save "gradients-iter-$(iterString).bson" meanLossGradients
        check_file("gradients-iter-$(iterString).bson")

        # Update the model
        update_model!(model, optimizer, meanLossGradients)

        # Move on to the next iteration
        iteration += 1
    end
    println("The training is finished!")
end

# -----------------------------------------------------------------------------
# --- Simulation

function simulate!(model, global_params, mc_params, nn_params, system_params)
    # Pack inputs
    input = MonteCarloSampleInput(global_params, mc_params, nn_params, system_params, model)
    inputs = [input for worker in workers()]

    # Run the simulation in parallel
    outputs = pmap(mcsample!, inputs)

    # Collect averages corresponding to each reference system
    systemOutputs, systemLosses = collect_system_averages(outputs, nothing, [system_params], global_params, nothing,
                                                          nothing)

    # Write descriptors and energies
    name = system_params.systemName
    write_rdf("RDFNN-$(name).dat", systemOutputs[1].descriptor, system_params)
    write_energies("energies-$(name).dat", systemOutputs[1].energies, mc_params, system_params, 1)
    return
end

# -----------------------------------------------------------------------------
# --- Symmetry Function

function combine_symmetry_matrices(g2_matrix, g3_matrix, g9_matrix)
    if isempty(g3_matrix) && isempty(g9_matrix)
        return g2_matrix
    end

    matrices = [g2_matrix, g3_matrix, g9_matrix]
    return hcat(filter(!isempty, matrices)...)
end

function compute_g2_element(distance::T,
                            η::T,
                            r_cutoff::T,
                            r_shift::T)::T where {T <: AbstractFloat}
    distance <= zero(T) && return zero(T)

    shifted_distance = distance - r_shift
    exponential_term = exp(-η * shifted_distance^2)
    cutoff_term = distance_cutoff(distance, r_cutoff)

    return exponential_term * cutoff_term
end

function compute_g2(distances::AbstractVector{T},
                    η::T,
                    r_cutoff::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    acc = zero(T)
    @simd for i in eachindex(distances)
        @inbounds acc += compute_g2_element(distances[i], η, r_cutoff, r_shift)
    end
    return acc
end

function build_g2_matrix(distance_matrix::AbstractMatrix{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g2_functions = length(nn_params.G2Functions)
    g2_matrix = Matrix{T}(undef, n_atoms, n_g2_functions)

    for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]
        for (j, g2_func) in enumerate(nn_params.G2Functions)
            g2_matrix[i, j] = compute_g2(distance_vector,
                                         g2_func.eta,
                                         g2_func.rcutoff,
                                         g2_func.rshift)
        end
    end

    return nn_params.symmFunctionScaling == one(T) ? g2_matrix :
           g2_matrix .* nn_params.symmFunctionScaling
end

function update_g2_matrix!(g2_matrix::AbstractMatrix{T},
                           distance_vector1::AbstractVector{T},
                           distance_vector2::AbstractVector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           point_index::Integer)::AbstractMatrix{T} where {T <: AbstractFloat}
    scaling = nn_params.symmFunctionScaling

    # Update displaced particle row
    @inbounds for (j, g2_func) in enumerate(nn_params.G2Functions)
        g2_matrix[point_index, j] = compute_g2(distance_vector2,
                                               g2_func.eta,
                                               g2_func.rcutoff,
                                               g2_func.rshift) * scaling
    end

    # Update affected atoms
    @inbounds for i in 1:(system_params.N)
        i == point_index && continue

        for (j, g2_func) in enumerate(nn_params.G2Functions)
            r_cutoff = g2_func.rcutoff
            dist1, dist2 = distance_vector1[i], distance_vector2[i]

            if (zero(T) < dist1 < r_cutoff) || (zero(T) < dist2 < r_cutoff)
                δg2 = compute_g2_element(dist2, g2_func.eta, r_cutoff, g2_func.rshift) -
                      compute_g2_element(dist1, g2_func.eta, r_cutoff, g2_func.rshift)
                g2_matrix[i, j] += δg2 * scaling
            end
        end
    end

    return g2_matrix
end

function compute_cos_angle(coordinates::AbstractMatrix{T},
                           box::AbstractVector{T},
                           i::Integer,
                           j::Integer,
                           k::Integer,
                           distance_ij::T,
                           distance_ik::T)::T where {T <: AbstractFloat}
    @assert i != j&&i != k && k != j "Indices must be different"

    atom_i = @view coordinates[:, i]
    vector_ij = compute_directional_vector(atom_i, @view(coordinates[:, j]), box)
    vector_ik = compute_directional_vector(atom_i, @view(coordinates[:, k]), box)

    cos_angle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)

    # -1 ≤ cos_angle ≤ 1
    cos_angle = clamp(cos_angle, -one(T), one(T))

    return cos_angle
end

function compute_triplet_geometry(coordinates::AbstractMatrix{T},
                                  box::AbstractVector{T},
                                  i::Integer,
                                  j::Integer,
                                  k::Integer,
                                  distance_ij::T,
                                  distance_ik::T)::Tuple{T, T} where {T <: AbstractFloat}
    @assert i != j&&i != k && k != j "Indices must be different"

    @inbounds begin
        atom_i = @view coordinates[:, i]
        atom_j = @view coordinates[:, j]
        atom_k = @view coordinates[:, k]

        distance_kj = compute_distance(atom_k, atom_j, box)
        vector_ij = compute_directional_vector(atom_i, atom_j, box)
        vector_ik = compute_directional_vector(atom_i, atom_k, box)

        cos_angle = dot(vector_ij, vector_ik) / (distance_ij * distance_ik)
        cos_angle = clamp(cos_angle, -one(T), one(T))
    end

    return cos_angle, distance_kj
end

function compute_g3_element(cos_angle::T,
                            distance_ij::T,
                            distance_ik::T,
                            distance_kj::T,
                            r_cutoff::T,
                            η::T,
                            ζ::T,
                            λ::T,
                            r_shift::T)::T where {T <: AbstractFloat}
    @fastmath begin
        angle_term = (one(T) + λ * cos_angle)^ζ

        squared_distances = (distance_ij - r_shift)^2 +
                            (distance_ik - r_shift)^2 +
                            (distance_kj - r_shift)^2
        radial_term = exp(-η * squared_distances)

        cutoff_term = distance_cutoff(distance_ij, r_cutoff) *
                      distance_cutoff(distance_ik, r_cutoff) *
                      distance_cutoff(distance_kj, r_cutoff)

        return angle_term * radial_term * cutoff_term
    end
end

function compute_g3(atom_index::Integer,
                    coordinates::AbstractMatrix{T},
                    box::AbstractVector{T},
                    distance_vector::AbstractVector{T},
                    r_cutoff::T,
                    η::T,
                    ζ::T,
                    λ::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    accumulator = zero(T)
    norm_factor = T(2)^(one(T) - ζ)

    @inbounds for k in eachindex(distance_vector)
        distance_ik = distance_vector[k]
        distance_ik <= zero(T) && continue

        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distance_vector[j]

            if zero(T) < distance_ij < r_cutoff && distance_ik < r_cutoff
                cos_angle, distance_kj = compute_triplet_geometry(coordinates, box, atom_index, j, k, distance_ij,
                                                                  distance_ik)

                contribution = compute_g3_element(cos_angle, distance_ij, distance_ik, distance_kj,
                                                  r_cutoff, η, ζ, λ, r_shift)

                accumulator += contribution
            end
        end
    end

    return norm_factor * accumulator
end

function build_g3_matrix(distance_matrix::AbstractMatrix{T},
                         coordinates::AbstractMatrix{T},
                         box::AbstractVector{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g3_functions = length(nn_params.G3Functions)
    g3_matrix = Matrix{T}(undef, n_atoms, n_g3_functions)

    @inbounds for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]

        for (j, g3_func) in enumerate(nn_params.G3Functions)
            g3_matrix[i, j] = compute_g3(i, coordinates, box, distance_vector,
                                         g3_func.rcutoff, g3_func.eta,
                                         g3_func.zeta, g3_func.lambda,
                                         g3_func.rshift)
        end
    end

    scaling = nn_params.symmFunctionScaling
    return scaling == one(T) ? g3_matrix : rmul!(g3_matrix, scaling)
end

function updateG3Matrix!(G3Matrix, coordinates1, coordinates2, box, distanceVector1, distanceVector2, systemParms,
                         NNParms, displacedAtomIndex)
    for selectedAtomIndex in 1:(systemParms.N)
        # Rebuild the whole G3 matrix column for the displaced atom
        if selectedAtomIndex == displacedAtomIndex
            for (G3Index, G3Function) in enumerate(NNParms.G3Functions)
                eta = G3Function.eta
                lambda = G3Function.lambda
                zeta = G3Function.zeta
                rcutoff = G3Function.rcutoff
                rshift = G3Function.rshift
                G3Matrix[selectedAtomIndex, G3Index] = compute_g3(displacedAtomIndex, coordinates2, box,
                                                                  distanceVector2,
                                                                  rcutoff, eta, zeta, lambda, rshift) *
                                                       NNParms.symmFunctionScaling
            end
            # Compute the change in G3 caused by the displacement of an atom
            # New ijk triplet description
            # Central atom (i): selectedAtomIndex
            # Second atom (j): displacedAtomIndex
            # Third atom (k): thirdAtomIndex
        else
            for (G3Index, G3Function) in enumerate(NNParms.G3Functions)
                rcutoff = G3Function.rcutoff
                distance_ij_1 = distanceVector1[selectedAtomIndex]
                distance_ij_2 = distanceVector2[selectedAtomIndex]
                # Ignore if the selected atom is far from the displaced atom
                if 0.0 < distance_ij_2 < rcutoff || 0.0 < distance_ij_1 < rcutoff
                    eta = G3Function.eta
                    lambda = G3Function.lambda
                    zeta = G3Function.zeta
                    rshift = G3Function.rshift
                    # Accumulate differences for the selected atom
                    # over all third atoms
                    ΔG3 = 0.0
                    # Loop over all ik pairs
                    for thirdAtomIndex in 1:(systemParms.N)
                        # Make sure i != j != k
                        if thirdAtomIndex != displacedAtomIndex && thirdAtomIndex != selectedAtomIndex
                            # It does not make a difference whether
                            # coordinates2 or coordinates1 are used -
                            # both selectedAtom and thirdAtom have
                            # have the same coordinates in the old and
                            # the updated configuration
                            selectedAtom = coordinates2[:, selectedAtomIndex]
                            thirdAtom = coordinates2[:, thirdAtomIndex]
                            distance_ik = compute_distance(selectedAtom, thirdAtom, box)
                            # The current ik pair is fixed so if r_ik > rcutoff
                            # no change in this G3(i,j,k) is needed
                            if 0.0 < distance_ik < rcutoff
                                # Compute the contribution to the change
                                # from the old configuration
                                displacedAtom_1 = coordinates1[:, displacedAtomIndex]
                                displacedAtom_2 = coordinates2[:, displacedAtomIndex]
                                distance_kj_1 = compute_distance(displacedAtom_1, thirdAtom, box)
                                distance_kj_2 = compute_distance(displacedAtom_2, thirdAtom, box)
                                if 0.0 < distance_kj_1 < rcutoff || 0.0 < distance_kj_2 < rcutoff
                                    # Compute cos of angle
                                    vector_ij_1 = compute_directional_vector(selectedAtom, displacedAtom_1, box)
                                    vector_ij_2 = compute_directional_vector(selectedAtom, displacedAtom_2, box)
                                    vector_ik = compute_directional_vector(selectedAtom, thirdAtom, box)
                                    cosAngle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                    cosAngle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)
                                    @assert -1.0 <= cosAngle1 <= 1.0
                                    @assert -1.0 <= cosAngle2 <= 1.0
                                    # Compute differences in G3
                                    G3_1 = compute_g3_element(cosAngle1, distance_ij_1, distance_ik, distance_kj_1,
                                                              rcutoff, eta, zeta, lambda, rshift)
                                    G3_2 = compute_g3_element(cosAngle2, distance_ij_2, distance_ik, distance_kj_2,
                                                              rcutoff, eta, zeta, lambda, rshift)
                                    ΔG3 += 2.0^(1.0 - zeta) * (G3_2 - G3_1)
                                end
                            end
                        end
                    end
                    # Apply the computed differences
                    G3Matrix[selectedAtomIndex, G3Index] += ΔG3 * NNParms.symmFunctionScaling
                end
            end
        end
    end
    return (G3Matrix)
end

function compute_g9_element(cos_angle::T,
                            distance_ij::T,
                            distance_ik::T,
                            r_cutoff::T,
                            η::T,
                            ζ::T,
                            λ::T,
                            r_shift::T)::T where {T <: AbstractFloat}
    @fastmath begin
        angle_term = (one(T) + λ * cos_angle)^ζ

        r_ij = distance_ij - r_shift
        r_ik = distance_ik - r_shift
        radial_term = exp(-η * (r_ij^2 + r_ik^2))

        cutoff_term = distance_cutoff(distance_ij, r_cutoff) *
                      distance_cutoff(distance_ik, r_cutoff)

        return angle_term * radial_term * cutoff_term
    end
end

function compute_g9(i, coordinates, box, distanceVector, rcutoff, eta, zeta, lambda, rshift)::Float64
    sum = 0.0
    @inbounds for k in eachindex(distanceVector)
        distance_ik = distanceVector[k]
        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distanceVector[j]
            if 0 < distance_ij < rcutoff && 0 < distance_ik < rcutoff
                cosAngle = compute_cos_angle(coordinates, box, i, j, k, distance_ij, distance_ik)
                sum += compute_g9_element(cosAngle, distance_ij, distance_ik, rcutoff, eta, zeta, lambda, rshift)
            end
        end
    end
    return (2.0^(1.0 - zeta) * sum)
end

function compute_g9(atom_index::Integer,
                    coordinates::AbstractMatrix{T},
                    box::AbstractVector{T},
                    distance_vector::AbstractVector{T},
                    r_cutoff::T,
                    η::T,
                    ζ::T,
                    λ::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    accumulator = zero(T)
    norm_factor = T(2)^(one(T) - ζ)

    @inbounds for k in eachindex(distance_vector)
        distance_ik = distance_vector[k]
        distance_ik <= zero(T) && continue

        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distance_vector[j]

            if zero(T) < distance_ij < r_cutoff && distance_ik < r_cutoff
                cos_angle = compute_cos_angle(coordinates, box, atom_index, j, k,
                                              distance_ij, distance_ik)

                accumulator += compute_g9_element(cos_angle, distance_ij, distance_ik,
                                                  r_cutoff, η, ζ, λ, r_shift)
            end
        end
    end

    return norm_factor * accumulator
end

function build_g9_matrix(distance_matrix::AbstractMatrix{T},
                         coordinates::AbstractMatrix{T},
                         box::AbstractVector{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g9_functions = length(nn_params.G9Functions)
    g9_matrix = Matrix{T}(undef, n_atoms, n_g9_functions)

    @inbounds for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]

        for (j, g9_func) in enumerate(nn_params.G9Functions)
            g9_matrix[i, j] = compute_g9(i, coordinates, box, distance_vector,
                                         g9_func.rcutoff, g9_func.eta,
                                         g9_func.zeta, g9_func.lambda,
                                         g9_func.rshift)
        end
    end

    scaling = nn_params.symmFunctionScaling
    return scaling == one(T) ? g9_matrix : rmul!(g9_matrix, scaling)
end

function updateG9Matrix!(G9Matrix, coordinates1, coordinates2, box, distanceVector1, distanceVector2, systemParms,
                         NNParms, displacedAtomIndex)
    for selectedAtomIndex in 1:(systemParms.N)
        # Rebuild the whole G9 matrix column for the displaced atom
        if selectedAtomIndex == displacedAtomIndex
            for (G9Index, G9Function) in enumerate(NNParms.G9Functions)
                eta = G9Function.eta
                lambda = G9Function.lambda
                zeta = G9Function.zeta
                rcutoff = G9Function.rcutoff
                rshift = G9Function.rshift
                G9Matrix[selectedAtomIndex, G9Index] = compute_g9(displacedAtomIndex, coordinates2, box,
                                                                  distanceVector2,
                                                                  rcutoff, eta, zeta, lambda, rshift) *
                                                       NNParms.symmFunctionScaling
            end
            # Compute the change in G9 caused by the displacement of an atom
            # New ijk triplet description
            # Central atom (i): selectedAtomIndex
            # Second atom (j): displacedAtomIndex
            # Third atom (k): thirdAtomIndex
        else
            for (G9Index, G9Function) in enumerate(NNParms.G9Functions)
                rcutoff = G9Function.rcutoff
                distance_ij_1 = distanceVector1[selectedAtomIndex]
                distance_ij_2 = distanceVector2[selectedAtomIndex]
                # Ignore if the selected atom is far from the displaced atom
                if 0.0 < distance_ij_2 < rcutoff || 0.0 < distance_ij_1 < rcutoff
                    eta = G9Function.eta
                    lambda = G9Function.lambda
                    zeta = G9Function.zeta
                    rshift = G9Function.rshift
                    # Accumulate differences for the selected atom
                    # over all third atoms
                    ΔG9 = 0.0
                    # Loop over all ik pairs
                    for thirdAtomIndex in 1:(systemParms.N)
                        # Make sure i != j != k
                        if thirdAtomIndex != displacedAtomIndex && thirdAtomIndex != selectedAtomIndex
                            # It does not make a difference whether
                            # coordinates2 or coordinates1 are used -
                            # both selectedAtom and thirdAtom have
                            # have the same coordinates in the old and
                            # the updated configuration
                            selectedAtom = coordinates2[:, selectedAtomIndex]
                            thirdAtom = coordinates2[:, thirdAtomIndex]
                            distance_ik = compute_distance(selectedAtom, thirdAtom, box)
                            # The current ik pair is fixed so if r_ik > rcutoff
                            # no change in this G9(i,j,k) is needed
                            if 0.0 < distance_ik < rcutoff
                                # Compute the contribution to the change
                                # from the old configuration
                                displacedAtom_1 = coordinates1[:, displacedAtomIndex]
                                displacedAtom_2 = coordinates2[:, displacedAtomIndex]
                                # Compute cos of angle
                                vector_ij_1 = compute_directional_vector(selectedAtom, displacedAtom_1, box)
                                vector_ij_2 = compute_directional_vector(selectedAtom, displacedAtom_2, box)
                                vector_ik = compute_directional_vector(selectedAtom, thirdAtom, box)
                                cosAngle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                cosAngle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)
                                @assert -1.0 <= cosAngle1 <= 1.0
                                @assert -1.0 <= cosAngle2 <= 1.0
                                # Compute differences in G9
                                G9_1 = compute_g9_element(cosAngle1, distance_ij_1, distance_ik, rcutoff, eta, zeta,
                                                          lambda, rshift)
                                G9_2 = compute_g9_element(cosAngle2, distance_ij_2, distance_ik, rcutoff, eta, zeta,
                                                          lambda, rshift)
                                ΔG9 += 2.0^(1.0 - zeta) * (G9_2 - G9_1)
                            end
                        end
                    end
                    # Apply the computed differences
                    G9Matrix[selectedAtomIndex, G9Index] += ΔG9 * NNParms.symmFunctionScaling
                end
            end
        end
    end
    return (G9Matrix)
end

# -----------------------------------------------------------------------------
# --- Monte Carlo

function update_distance_histogram!(distance_matrix, histogram, system_params)
    n_atoms = system_params.N
    bin_width = system_params.binWidth
    n_bins = system_params.Nbins

    @inbounds for i in 1:n_atoms
        @fastmath for j in 1:(i - 1)
            bin_index = floor(Int, 1 + distance_matrix[i, j] / bin_width)
            if bin_index <= n_bins
                histogram[bin_index] += 1
            end
        end
    end

    return histogram
end

function update_distance_histogram_vectors!(histogram, old_distances, new_distances, system_params)
    n_atoms = system_params.N
    bin_width = system_params.binWidth
    n_bins = system_params.Nbins

    @inbounds @fastmath for i in 1:n_atoms
        # Remove old distance contribution
        old_bin = floor(Int, 1 + old_distances[i] / bin_width)
        if old_bin <= n_bins
            histogram[old_bin] -= 1
        end

        # Add new distance contribution
        new_bin = floor(Int, 1 + new_distances[i] / bin_width)
        if new_bin <= n_bins
            histogram[new_bin] += 1
        end
    end

    return histogram
end

function normalize_hist_to_rdf!(histogram::AbstractVector{T},
                                system_params::SystemParameters,
                                box::AbstractVector{T})::AbstractVector{T} where {T <: AbstractFloat}
    # Calculate system volumes and pair count
    box_volume = prod(box)
    n_pairs = system_params.N * (system_params.N - 1) ÷ 2

    # Pre-calculate bin radii and shell volumes
    bin_width = system_params.binWidth
    bins = [(i * bin_width) for i in 1:(system_params.Nbins)]
    shell_volumes = [4 * π * system_params.binWidth * bins[i]^2 for i in eachindex(bins)]

    # Calculate normalization factors
    normalization_factors = fill(box_volume / n_pairs, system_params.Nbins) ./ shell_volumes

    # Apply normalization in-place
    histogram .*= normalization_factors

    return histogram
end

function apply_periodic_boundaries!(frame::Frame,
                                    box::AbstractVector{T},
                                    point_index::Integer) where {T <: AbstractFloat}
    pos = positions(frame)

    @inbounds for dim in 1:3
        coord = pos[dim, point_index]
        box_length = box[dim]

        if coord < zero(T)
            pos[dim, point_index] += box_length
        elseif coord > box_length
            pos[dim, point_index] -= box_length
        end
    end

    return frame
end

function mcmove!(mcarrays, E, EpreviousVector, model, NNParms, systemParms, box, rng, mutatedStepAdjust)
    # Unpack mcarrays
    frame, distanceMatrix, G2Matrix1, G3Matrix1, G9Matrix1 = mcarrays
    # Optionally make a copy of the original coordinates
    if G3Matrix1 != [] || G9Matrix1 != []
        coordinates1 = copy(positions(frame))
    end

    # Pick a particle
    pointIndex = rand(rng, Int32(1):Int32(systemParms.N))

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Take a copy of the previous energy value
    E1 = copy(E)

    # Displace the particle
    dr = mutatedStepAdjust * (rand(rng, Float64, 3) .- 0.5)

    positions(frame)[:, pointIndex] .+= dr

    # Check if all coordinates inside simulation box with PBC
    apply_periodic_boundaries!(frame, box, pointIndex)

    # Compute the updated distance vector
    point = positions(frame)[:, pointIndex]
    distanceVector2 = compute_distance_vector(point, positions(frame), box)

    # Acceptance counter
    accepted = 0

    # Get indexes of atoms for energy contribution update
    indexesForUpdate = get_energies_update_mask(distanceVector2, NNParms)

    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(G2Matrix1)
    update_g2_matrix!(G2Matrix2, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

    # Make a copy of the original angular matrices and update them
    G3Matrix2 = copy(G3Matrix1)
    if G3Matrix1 != []
        updateG3Matrix!(G3Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    G9Matrix2 = copy(G9Matrix1)
    if G9Matrix1 != []
        updateG9Matrix!(G9Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Combine symmetry function matrices accumulators
    symmFuncMatrix2 = combine_symmetry_matrices(G2Matrix2, G3Matrix2, G9Matrix2)

    # Compute the energy again
    # E2 = totalEnergyScalar(G2Matrix2, model)
    newEnergyVector = update_system_energies_vector(symmFuncMatrix2, model, indexesForUpdate, EpreviousVector)
    E2 = sum(newEnergyVector)

    # Get energy difference
    ΔE = E2 - E1

    # Accept or reject the move
    if rand(rng, Float64) < exp(-ΔE * systemParms.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector2
        distanceMatrix[:, pointIndex] = distanceVector2
        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix2, G3Matrix2, G9Matrix2)
        return (mcarrays, E, newEnergyVector, accepted)
    else
        positions(frame)[:, pointIndex] .-= dr

        # Check if all coordinates inside simulation box with PBC
        apply_periodic_boundaries!(frame, box, pointIndex)

        # Pack mcarrays
        mcarrays = (frame, distanceMatrix, G2Matrix1, G3Matrix1, G9Matrix1)
        return (mcarrays, E, EpreviousVector, accepted)
    end
end

function mcsample!(input)
    # Unpack the inputs
    model = input.model
    globalParms = input.globalParms
    MCParms = input.MCParms
    NNParms = input.NNParms
    systemParms = input.systemParms

    mutatedStepAdjust = copy(systemParms.Δ)

    # Get the worker id and the output filenames
    if nprocs() == 1
        id = myid()
    else
        id = myid() - 1
    end
    idString = lpad(id, 3, '0')

    trajFile = "mctraj-p$(idString).xtc"
    pdbFile = "confin-p$(idString).pdb"

    # Initialize RNG
    rngXor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    if globalParms.mode == "training"
        # Take a random frame from the equilibrated trajectory
        traj = read_xtc(systemParms)
        nframes = Int(size(traj)) - 1
        frameId = rand(rngXor, 1:nframes) # Don't take the first frame
        frame = deepcopy(read_step(traj, frameId))
    else
        # Read PDB data from the system.in file
        pdb = read_pdb(systemParms)
        frame = deepcopy(read_step(pdb, 0))
    end

    # Get current box vectors
    box = lengths(UnitCell(frame))

    # Start writing MC trajectory
    if globalParms.outputMode == "verbose"
        write_trajectory(positions(frame), box, systemParms, trajFile, 'w')
        write_trajectory(positions(frame), box, systemParms, pdbFile, 'w')
    end

    # Get the number of data points
    totalDataPoints = Int(MCParms.steps / MCParms.outfreq)
    prodDataPoints = Int((MCParms.steps - MCParms.Eqsteps) / MCParms.outfreq)

    # Build the distance matrix
    distanceMatrix = build_distance_matrix(frame)

    # Build the symmetry function matrices
    G2Matrix = build_g2_matrix(distanceMatrix, NNParms)

    G3Matrix = []
    if length(NNParms.G3Functions) > 0
        G3Matrix = build_g3_matrix(distanceMatrix, positions(frame), box, NNParms)
    end

    G9Matrix = []
    if length(NNParms.G9Functions) > 0
        G9Matrix = build_g9_matrix(distanceMatrix, positions(frame), box, NNParms)
    end

    # Prepare a tuple of arrays that change duing the mcmove!
    mcarrays = (frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix)

    # Initialize the distance histogram accumulator
    histAccumulator = zeros(Float64, systemParms.Nbins)

    # Build the cross correlation arrays for training,
    # an additional distance histogram array
    # and the symmetry function matrix accumulator
    if globalParms.mode == "training"
        hist = zeros(Float64, systemParms.Nbins)
        G2MatrixAccumulator = zeros(size(G2Matrix))
        G3MatrixAccumulator = zeros(size(G3Matrix))
        G9MatrixAccumulator = zeros(size(G9Matrix))
        crossAccumulators = initialize_cross_accumulators(NNParms, systemParms)
    end

    # Combine symmetry function matrices
    symmFuncMatrix = combine_symmetry_matrices(G2Matrix, G3Matrix, G9Matrix)

    # Initialize the starting energy and the energy array
    EpreviousVector = init_system_energies_vector(symmFuncMatrix, model)
    E = sum(EpreviousVector)
    energies = zeros(totalDataPoints + 1)
    energies[1] = E

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @fastmath for step in 1:(MCParms.steps)
        mcarrays, E, EpreviousVector, accepted = mcmove!(mcarrays, E, EpreviousVector, model, NNParms, systemParms, box,
                                                         rngXor, mutatedStepAdjust)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        # Perform MC step adjustment during the equilibration
        if MCParms.stepAdjustFreq > 0 && step % MCParms.stepAdjustFreq == 0 && step < MCParms.Eqsteps
            mutatedStepAdjust = adjust_monte_carlo_step!(mutatedStepAdjust, systemParms, box, MCParms,
                                                         acceptedIntermediate)
            acceptedIntermediate = 0
        end

        # Collect the output energies
        if step % MCParms.outfreq == 0
            energies[Int(step / MCParms.outfreq) + 1] = E
        end

        # MC trajectory output
        if globalParms.outputMode == "verbose"
            if step % MCParms.trajout == 0
                write_trajectory(positions(mcarrays[1]), box, systemParms, trajFile, 'a')
            end
        end

        # Accumulate the distance histogram
        if step % MCParms.outfreq == 0 && step > MCParms.Eqsteps
            frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix = mcarrays
            # Update the cross correlation array during the training
            if globalParms.mode == "training"
                hist = update_distance_histogram!(distanceMatrix, hist, systemParms)
                histAccumulator .+= hist
                G2MatrixAccumulator .+= G2Matrix
                if G3Matrix != []
                    G3MatrixAccumulator .+= G3Matrix
                end
                if G9Matrix != []
                    G9MatrixAccumulator .+= G9Matrix
                end
                # Normalize the histogram to RDF
                normalize_hist_to_rdf!(hist, systemParms, box)

                # Combine symmetry function matrices
                symmFuncMatrix = combine_symmetry_matrices(G2Matrix, G3Matrix, G9Matrix)

                update_cross_accumulators!(crossAccumulators, symmFuncMatrix, hist, model, NNParms)
                # Nullify the hist array for the next training iteration
                hist = zeros(Float64, systemParms.Nbins)
            else
                histAccumulator = update_distance_histogram!(distanceMatrix, histAccumulator, systemParms)
            end
        end
    end
    # Compute and report the final acceptance ratio
    acceptanceRatio = acceptedTotal / MCParms.steps

    # Unpack mcarrays and optionally normalize cross and G2Matrix accumulators
    frame, distanceMatrix, G2Matrix, G3Matrix, G9Matrix = mcarrays # might remove this line
    if globalParms.mode == "training"
        # Normalize the cross correlation arrays
        for cross in crossAccumulators
            cross ./= prodDataPoints
        end
        G2MatrixAccumulator ./= prodDataPoints
        if G3Matrix != []
            G3MatrixAccumulator ./= prodDataPoints
        end
        if G9Matrix != []
            G9MatrixAccumulator ./= prodDataPoints
        end
        symmFuncMatrixAccumulator = combine_symmetry_matrices(G2MatrixAccumulator, G3MatrixAccumulator,
                                                              G9MatrixAccumulator)
    end

    # Normalize the sampled distance histogram
    histAccumulator ./= prodDataPoints
    normalize_hist_to_rdf!(histAccumulator, systemParms, box)

    # Combine symmetry function matrices accumulators
    if globalParms.mode == "training"
        MCoutput = MonteCarloAverages(histAccumulator, energies, crossAccumulators, symmFuncMatrixAccumulator,
                                      acceptanceRatio,
                                      systemParms, mutatedStepAdjust)
        return (MCoutput)
    else
        MCoutput = MonteCarloAverages(histAccumulator, energies, nothing, nothing, acceptanceRatio, systemParms,
                                      mutatedStepAdjust)
        return (MCoutput)
    end
end

function adjust_monte_carlo_step!(current_step_size::T,
                                  system_params::SystemParameters,
                                  box::AbstractVector{T},
                                  mc_params::MonteCarloParameters,
                                  accepted_moves::Integer)::T where {T <: AbstractFloat}
    # Calculate current acceptance ratio
    acceptance_ratio = accepted_moves / mc_params.stepAdjustFreq

    # Adjust step size based on target acceptance ratio
    current_step_size = (acceptance_ratio / system_params.targetAR) * current_step_size

    # Limit maximum step size to half of smallest box dimension
    max_step_size = minimum(box) / 2
    if current_step_size > max_step_size
        current_step_size = max_step_size
    end

    return current_step_size
end

# -----------------------------------------------------------------------------
# --- Pre Training

function compute_pmf(rdf::AbstractVector{T},
                     system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_bins = system_params.Nbins
    β = system_params.β

    # Initialize PMF array and identify repulsive regions
    pmf = Vector{T}(undef, n_bins)
    repulsion_mask = rdf .== zero(T)
    n_repulsion_points = count(repulsion_mask)

    # Calculate reference PMF values
    first_valid_index = n_repulsion_points + 1
    max_pmf = -log(rdf[first_valid_index]) / β
    next_pmf = -log(rdf[first_valid_index + 1]) / β
    pmf_gradient = max_pmf - next_pmf

    # Fill PMF array
    @inbounds for i in eachindex(pmf)
        if repulsion_mask[i]
            # Extrapolate PMF in repulsive region
            pmf[i] = max_pmf + pmf_gradient * (first_valid_index - i)
        else
            # Calculate PMF from RDF
            pmf[i] = -log(rdf[i]) / β
        end
    end

    return pmf
end

function precompute_reference_data(input::PreComputedInput;)::ReferenceData
    # Unpack input parameters
    nn_params = input.NNParms
    system_params = input.systemParms
    ref_rdf = input.refRDF

    println("Pre-computing data for $(system_params.systemName)...")

    # Compute potential of mean force
    pmf = compute_pmf(ref_rdf, system_params)

    # Initialize trajectory reading
    trajectory = read_xtc(system_params)
    n_frames = Int(size(trajectory)) - 1  # Skip first frame

    # Pre-allocate data containers
    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    histograms = Vector{Vector{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    g3_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.G3Functions) > 0))
    g9_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.G9Functions) > 0))

    # Process each frame
    for frame_id in 1:n_frames
        frame = read_step(trajectory, frame_id)
        box = lengths(UnitCell(frame))
        coords = positions(frame)

        # Compute distance matrix and histogram
        distance_matrices[frame_id] = build_distance_matrix(frame)
        histograms[frame_id] = zeros(Float64, system_params.Nbins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], system_params)

        # Compute symmetry function matrices
        g2_matrices[frame_id] = build_g2_matrix(distance_matrices[frame_id], nn_params)

        if !isempty(nn_params.G3Functions)
            g3_matrices[frame_id] = build_g3_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end

        if !isempty(nn_params.G9Functions)
            g9_matrices[frame_id] = build_g9_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end
    end

    return ReferenceData(distance_matrices,
                         histograms,
                         pmf,
                         g2_matrices,
                         g3_matrices,
                         g9_matrices)
end

function compute_pretraining_gradient(energy_diff_nn::T,
                                      energy_diff_pmf::T,
                                      symm_func_matrix1::AbstractMatrix{T},
                                      symm_func_matrix2::AbstractMatrix{T},
                                      model::Chain,
                                      pretrain_params::PreTrainingParameters,
                                      nn_params::NeuralNetParameters;
                                      log_file::String="pretraining-loss-values.out",
                                      verbose::Bool=false) where {T <: AbstractFloat}
    # Calculate MSE loss between energy differences
    mse_loss = (energy_diff_nn - energy_diff_pmf)^2

    # Log loss value
    try
        open(log_file, "a") do io
            println(io, round(mse_loss; digits=8))
        end
    catch e
        @warn "Failed to write to log file: $log_file" exception=e
    end

    # Calculate regularization loss
    model_params = Flux.params(model)
    reg_loss = pretrain_params.PTREGP * sum(abs2, model_params[1])

    # Print detailed loss information if requested
    if verbose
        println("""
                  Energy loss: $(round(mse_loss; digits=8))
                  PMF energy difference: $(round(energy_diff_pmf; digits=8))
                  NN energy difference: $(round(energy_diff_nn; digits=8))
                  Regularization loss: $(round(reg_loss; digits=8))
                """)
    end

    # Compute gradients for both configurations
    grad1 = compute_energy_gradients(symm_func_matrix1, model, nn_params)
    grad2 = compute_energy_gradients(symm_func_matrix2, model, nn_params)

    # Calculate loss gradients with regularization
    gradient_scale = 2 * (energy_diff_nn - energy_diff_pmf)
    loss_gradient = @. gradient_scale * (grad2 - grad1)
    reg_gradient = @. model_params * 2 * pretrain_params.PTREGP

    return loss_gradient + reg_gradient
end

function pretraining_move!(refData::ReferenceData, model, NNParms, systemParms, rng)
    # Pick a frame
    traj = read_xtc(systemParms)
    nframes = Int(size(traj)) - 1 # Don't take the first frame
    frameId::Int = rand(rng, 1:nframes)
    frame = read_step(traj, frameId)
    box = lengths(UnitCell(frame))
    # Pick a particle
    pointIndex::Int = rand(rng, 1:(systemParms.N))

    # Read reference data
    distanceMatrix = refData.distanceMatrices[frameId]
    hist = refData.histograms[frameId]
    PMF = refData.PMF

    # If no angular symmetry functions are provided, use G2 only
    if refData.G3Matrices == [] && refData.G9Matrices == []
        symmFuncMatrix1 = refData.G2Matrices[frameId]
    else
        # Make a copy of the original coordinates
        coordinates1 = copy(positions(frame))
        # Combine symmetry function matrices
        if refData.G3Matrices == []
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G9Matrices[frameId]]
        elseif refData.G9Matrices == []
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G3Matrices[frameId]]
        else
            symmFuncMatrices = [refData.G2Matrices[frameId], refData.G3Matrices[frameId], refData.G9Matrices[frameId]]
        end
        # Unpack symmetry functions and concatenate horizontally into a single matrix
        symmFuncMatrix1 = hcat(symmFuncMatrices...)
    end

    # Compute energy of the initial configuration
    ENN1Vector = init_system_energies_vector(symmFuncMatrix1, model)
    ENN1 = sum(ENN1Vector)
    EPMF1 = sum(hist .* PMF)

    # Allocate the distance vector
    distanceVector1 = distanceMatrix[:, pointIndex]

    # Displace the particle
    dr = systemParms.Δ * (rand(rng, Float64, 3) .- 0.5)

    positions(frame)[:, pointIndex] .+= dr

    # Compute the updated distance vector
    point = positions(frame)[:, pointIndex]
    distanceVector2 = compute_distance_vector(point, positions(frame), box)

    # Update the histogram
    hist = update_distance_histogram_vectors!(hist, distanceVector1, distanceVector2, systemParms)

    # Get indexes for updating ENN
    indexesForUpdate = get_energies_update_mask(distanceVector2, NNParms)

    # Make a copy of the original G2 matrix and update it
    G2Matrix2 = copy(refData.G2Matrices[frameId])
    update_g2_matrix!(G2Matrix2, distanceVector1, distanceVector2, systemParms, NNParms, pointIndex)

    # Make a copy of the original angular matrices and update them
    G3Matrix2 = []
    if refData.G3Matrices != []
        G3Matrix2 = copy(refData.G3Matrices[frameId])
        updateG3Matrix!(G3Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Make a copy of the original angular matrices and update them
    G9Matrix2 = []
    if refData.G9Matrices != []
        G9Matrix2 = copy(refData.G9Matrices[frameId])
        updateG9Matrix!(G9Matrix2, coordinates1, positions(frame), box, distanceVector1, distanceVector2, systemParms,
                        NNParms, pointIndex)
    end

    # Combine symmetry function matrices accumulators
    symmFuncMatrix2 = combine_symmetry_matrices(G2Matrix2, G3Matrix2, G9Matrix2)

    # Compute the NN energy again
    ENN2Vector = update_system_energies_vector(symmFuncMatrix2, model, indexesForUpdate, ENN1Vector)
    ENN2 = sum(ENN2Vector)
    EPMF2 = sum(hist .* PMF)

    # Get the energy differences
    ΔENN = ENN2 - ENN1
    ΔEPMF = EPMF2 - EPMF1

    # Revert the changes in the frame arrays
    positions(frame)[:, pointIndex] .-= dr
    hist = update_distance_histogram_vectors!(hist, distanceVector2, distanceVector1, systemParms)

    return (symmFuncMatrix1, symmFuncMatrix2, ΔENN, ΔEPMF)
end

function pretrain_model!(pretrain_params::PreTrainingParameters,
                         nn_params::NeuralNetParameters,
                         system_params_list,
                         model::Chain,
                         optimizer::Flux.Optimise.AbstractOptimiser,
                         reference_rdfs;
                         save_path::String="model-pre-trained.bson",
                         verbose::Bool=true)::Chain
    if verbose
        println("\nStarting pre-training with Monte Carlo for $(pretrain_params.PTsteps) steps")
        println("Regularization parameter: $(pretrain_params.PTREGP)")
        report_optimizer(optimizer)
    end

    # Initialize random number generator and prepare reference data
    rng = RandomNumbers.Xorshifts.Xoroshiro128Plus()
    n_systems = length(system_params_list)

    # Prepare inputs for parallel computation
    ref_data_inputs = [PreComputedInput(nn_params, system_params_list[i], reference_rdfs[i])
                       for i in 1:n_systems]

    # Pre-compute reference data in parallel
    ref_data_list = pmap(precompute_reference_data, ref_data_inputs)

    # Main training loop
    for step in 1:(pretrain_params.PTsteps)
        should_report = (step % pretrain_params.PToutfreq == 0) || (step == 1)
        if should_report && verbose
            println("\nPre-training step: $step")
        end

        # Compute gradients for all systems
        loss_gradients = Vector{Any}(undef, n_systems)
        for sys_id in 1:n_systems
            ref_data = ref_data_list[sys_id]

            # Get energy differences and matrices for current system
            symm_func_matrix1, symm_func_matrix2, Δe_nn, Δe_pmf = pretraining_move!(ref_data,
                                                                                    model,
                                                                                    nn_params,
                                                                                    system_params_list[sys_id],
                                                                                    rng)

            # Calculate gradient for current system
            loss_gradients[sys_id] = compute_pretraining_gradient(Δe_nn,
                                                                  Δe_pmf,
                                                                  symm_func_matrix1,
                                                                  symm_func_matrix2,
                                                                  model,
                                                                  pretrain_params,
                                                                  nn_params,
                                                                  verbose=should_report)
        end

        # Update model with mean gradients
        mean_gradients = mean(loss_gradients)
        update_model!(model, optimizer, mean_gradients)
    end

    # Save trained model
    try
        @save save_path model
        check_file(save_path)
        verbose && println("\nModel saved to $save_path")
    catch e
        @warn "Failed to save model to $save_path" exception=e
    end

    return model
end

function main()
    # Initialize timing
    start_time = now()
    println("Starting at: ", start_time)

    # Initialize parameters and validate configuration
    global_params, mc_params, nn_params, pretrain_params, system_params_list = parameters_init()

    # Validate worker/system configuration
    num_workers = nworkers()
    num_systems = length(system_params_list)

    if num_workers % num_systems != 0
        throw(ArgumentError("Number of workers ($num_workers) must be divisible by number of systems ($num_systems)"))
    end

    # Initialize input data and model components
    inputs = input_init(global_params, nn_params, pretrain_params, system_params_list)
    model, optimizer, ref_rdfs = if global_params.mode == "training"
        inputs
    else
        inputs, nothing, nothing
    end

    # Display model configuration
    println("Using the following symmetry functions as the neural input for each atom:")
    print_symmetry_function_info(nn_params)

    # Execute workflow based on mode
    if global_params.mode == "training"
        println("""
            Training Configuration:
            - Using $(num_systems) reference system(s)
            - Activation functions: $(nn_params.activations)
            """)

        # Execute pretraining if needed
        if global_params.modelFile == "none"
            model = pretrain_model!(pretrain_params, nn_params, system_params_list, model, optimizer, ref_rdfs)

            println("\nRe-initializing the optimizer for the training...")
            optimizer = init_optimizer(nn_params)
            report_optimizer(optimizer)
            println("Neural network regularization parameter: $(nn_params.REGP)")
        end

        # Execute main training
        println("""
            \nStarting main training phase:
            - Adaptive gradient scaling: $(global_params.adaptiveScaling)
            - Iterations: $(nn_params.iters)
            - Running on $(num_workers) worker(s)
            - Total steps: $(mc_params.steps * num_workers / 1e6)M
            - Equilibration steps per rank: $(mc_params.Eqsteps / 1e6)M
            """)

        train!(global_params, mc_params, nn_params, system_params_list, model, optimizer, ref_rdfs)
    else
        length(system_params_list) == 1 || throw(ArgumentError("Simulation mode supports only one system"))
        println("Running simulation with $(global_params.modelFile)")
        simulate!(model, global_params, mc_params, nn_params, system_params_list[1])
    end

    # Log execution summary
    stop_time = now()
    wall_time = canonicalize(stop_time - start_time)
    println("\nExecution completed:")
    println("- Stop time: ", stop_time)
    println("- Wall time: ", wall_time)
end

end # module MlImc

if abspath(PROGRAM_FILE) == @__FILE__
    MlImc.main()
end
