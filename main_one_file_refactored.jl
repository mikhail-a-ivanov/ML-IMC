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
using TOML

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
    # Symmetry functions
    g2_functions::Vector{G2}
    g3_functions::Vector{G3}
    g9_functions::Vector{G9}
    max_distance_cutoff::Float64
    symm_function_scaling::Float64

    # Architecture
    neurons::Vector{Int}
    activations::Vector{String}
    bias::Bool

    # Training parameters
    iterations::Int
    regularization::Float64

    # Optimizer settings
    optimizer::String
    learning_rate::Float64
    momentum::Float64
    decay_1::Float64
    decay_2::Float64
end

struct SystemParameters
    system_name::String
    topology_file::String
    trajectory_file::String
    rdf_file::String
    n_atoms::Int
    atom_name::String
    n_bins::Int
    bin_width::Float64
    temperature::Float64
    beta::Float64
    max_displacement::Float64
    target_acceptance_ratio::Float64
end

struct MonteCarloParameters
    steps::Int
    equilibration_steps::Int
    step_adjust_frequency::Int
    trajectory_output_frequency::Int
    output_frequency::Int
end

struct GlobalParameters
    system_files::Vector{String}
    symmetry_function_file::String
    mode::String
    output_mode::String
    model_file::String
    gradients_file::String
    optimizer_file::String
    adaptive_scaling::Bool
end

struct PreTrainingParameters
    steps::Int
    output_frequency::Int
    regularization::Float64
    optimizer::String
    learning_rate::Float64
    momentum::Float64
    decay_1::Float64
    decay_2::Float64
end

struct MonteCarloSampleInput
    global_params::GlobalParameters
    mc_params::MonteCarloParameters
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    model::Chain
end

struct PreComputedInput
    nn_params::NeuralNetParameters
    system_params::SystemParameters
    reference_rdf::Vector{Float64}
end

struct MonteCarloAverages
    descriptor::Vector{Float64}
    energies::Vector{Float64}
    cross_accumulators::Union{Nothing, Vector{Matrix{Float64}}}
    symmetry_matrix_accumulator::Union{Nothing, Matrix{Float64}}
    acceptance_ratio::Float64
    system_params::SystemParameters
    step_size::Float64
end

struct ReferenceData
    distance_matrices::Vector{Matrix{Float64}}
    histograms::Vector{Vector{Float64}}
    pmf::Vector{Float64}
    g2_matrices::Vector{Matrix{Float64}}
    g3_matrices::Vector{Matrix{Float64}}
    g9_matrices::Vector{Matrix{Float64}}
end

# -----------------------------------------------------------------------------
# --- Helper Functions (Utils)

function report_optimizer(optimizer::Flux.Optimise.AbstractOptimiser)
    println("Optimizer type: $(typeof(optimizer))")
    println("   Parameters:")

    param_descriptions = Dict(:eta => "Learning rate",
                              :beta => "Decays",
                              :velocity => "Velocity",
                              :rho => "Momentum coefficient")

    for name in fieldnames(typeof(optimizer))
        if name ∉ (:state, :velocity)
            value = getfield(optimizer, name)
            description = get(param_descriptions, name, string(name))
            println("       $description: $value")
        end
    end
end

function print_symmetry_function_info(nn_params::NeuralNetParameters)
    for (func_type, functions) in [("G2", nn_params.g2_functions),
        ("G3", nn_params.g3_functions),
        ("G9", nn_params.g9_functions)]
        if !isempty(functions)
            println("    $func_type symmetry functions:")
            println("    eta, Å^-2; rcutoff, Å; rshift, Å")
            for func in functions
                println("       ", func)
            end
        end
    end

    println("Maximum cutoff distance: $(nn_params.max_distance_cutoff) Å")
    println("Symmetry function scaling parameter: $(nn_params.symm_function_scaling)")
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
    check_file(system_params.trajectory_file)
    return Trajectory(system_params.trajectory_file)
end

function read_pdb(system_params::SystemParameters)
    check_file(system_params.topology_file)
    return Trajectory(system_params.topology_file)
end

function write_rdf(outname::AbstractString, rdf::Vector{Float64}, system_params::SystemParameters)
    bins = [bin * system_params.bin_width for bin in 1:(system_params.n_bins)]

    open(outname, "w") do io
        println(io, "# System: $(system_params.system_name)")
        println(io, "# RDF data ($(system_params.atom_name) - $(system_params.atom_name))")
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
    steps = 0:(mc_params.output_frequency * slicing):(mc_params.steps)
    sliced_energies = energies[1:slicing:end]

    open(outname, "w") do io
        println(io, "# System: $(system_params.system_name)")
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

    for i in 1:(system_params.n_atoms)
        wrapped_atom_coords = wrap!(UnitCell(frame), view(conf, :, i)) .+ box_center
        add_atom!(frame, Atom(system_params.atom_name), wrapped_atom_coords)
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

function parse_symmetry_functions(filename::String)
    symm_data = TOML.parsefile(filename)

    g2_functions = Vector{G2}()
    g3_functions = Vector{G3}()
    g9_functions = Vector{G9}()
    scaling_factor = get(symm_data, "scaling", 1.0)
    max_cutoff = 0.0

    # Parse G2 functions
    if haskey(symm_data, "G2")
        for g2_params in symm_data["G2"]
            g2 = G2(g2_params["eta"],
                    g2_params["rcutoff"],
                    g2_params["rshift"])
            push!(g2_functions, g2)
            max_cutoff = max(max_cutoff, g2.rcutoff)
        end
    end

    # Parse G3 functions
    if haskey(symm_data, "G3")
        for g3_params in symm_data["G3"]
            g3 = G3(g3_params["eta"],
                    g3_params["lambda"],
                    g3_params["zeta"],
                    g3_params["rcutoff"],
                    g3_params["rshift"])
            push!(g3_functions, g3)
            max_cutoff = max(max_cutoff, g3.rcutoff)
        end
    end

    # Parse G9 functions
    if haskey(symm_data, "G9")
        for g9_params in symm_data["G9"]
            g9 = G9(g9_params["eta"],
                    g9_params["lambda"],
                    g9_params["zeta"],
                    g9_params["rcutoff"],
                    g9_params["rshift"])
            push!(g9_functions, g9)
            max_cutoff = max(max_cutoff, g9.rcutoff)
        end
    end

    return g2_functions, g3_functions, g9_functions, max_cutoff, scaling_factor
end

function parse_system_parameters(filename::String)
    # Constants
    NA = 6.02214076e23  # Avogadro constant
    kB = 1.38064852e-23 * NA / 1000  # Boltzmann constant in kJ/(mol·K)

    system_data = TOML.parsefile(filename)["system"]

    # Read and validate topology file
    pdb = Trajectory(system_data["topology_file_path"])
    frame = read(pdb)
    n_atoms = length(frame)
    atom_name = name(Atom(frame, 1))

    # Read RDF data
    bins, rdf = read_rdf(system_data["rdf_file_path"])
    n_bins = length(bins)
    bin_width = bins[1]

    # Calculate beta from temperature
    temperature = system_data["temperature"]
    beta = 1 / (kB * temperature)

    return SystemParameters(system_data["system_name"],
                            system_data["topology_file_path"],
                            system_data["trajectory_file_path"],
                            system_data["rdf_file_path"],
                            n_atoms,
                            atom_name,
                            n_bins,
                            bin_width,
                            temperature,
                            beta,
                            system_data["max_displacement"],
                            system_data["target_acceptance_ratio"])
end

function parameters_init()
    # Handle command line arguments
    inputname = if length(ARGS) > 0 && !occursin("json", ARGS[1])
        ARGS[1]
    else
        println("No input file was provided!")
        println("Trying to read input data from ML-IMC-init.toml")
        "ML-IMC-init.toml"
    end

    # Read and parse main configuration file
    config = TOML.parsefile(inputname)

    # Parse global parameters
    global_params = GlobalParameters(config["global"]["system_files"],
                                     config["global"]["symmetry_function_file"],
                                     config["global"]["mode"],
                                     config["global"]["output_mode"],
                                     config["global"]["model_file"],
                                     config["global"]["gradients_file"],
                                     config["global"]["optimizer_file"],
                                     config["global"]["adaptive_scaling"])

    # Parse Monte Carlo parameters
    mc_params = MonteCarloParameters(config["monte_carlo"]["steps"],
                                     config["monte_carlo"]["equilibration_steps"],
                                     config["monte_carlo"]["step_adjust_frequency"],
                                     config["monte_carlo"]["trajectory_output_frequency"],
                                     config["monte_carlo"]["output_frequency"])

    # Parse symmetry functions
    g2_funcs, g3_funcs, g9_funcs, max_cutoff, scaling = parse_symmetry_functions(global_params.symmetry_function_file)

    # Calculate input layer size based on total number of symmetry functions
    input_layer_size = length(g2_funcs) + length(g3_funcs) + length(g9_funcs)

    # Parse neural network parameters
    nn_section = config["neural_network"]

    # Prepend input layer size to existing neurons array
    neurons = [input_layer_size; nn_section["neurons"]]

    nn_params = NeuralNetParameters(g2_funcs,
                                    g3_funcs,
                                    g9_funcs,
                                    max_cutoff,
                                    scaling,
                                    neurons,
                                    nn_section["activations"],
                                    nn_section["bias"],
                                    nn_section["iterations"],
                                    nn_section["regularization"],
                                    nn_section["optimizer"],
                                    nn_section["learning_rate"],
                                    nn_section["momentum"],
                                    nn_section["decay_rates"][1],
                                    nn_section["decay_rates"][2])

    # Parse pre-training parameters
    pt_section = config["pretraining"]
    pretrain_params = PreTrainingParameters(pt_section["steps"],
                                            pt_section["output_frequency"],
                                            pt_section["regularization"],
                                            pt_section["optimizer"],
                                            pt_section["learning_rate"],
                                            pt_section["momentum"],
                                            pt_section["decay_rates"][1],
                                            pt_section["decay_rates"][2])

    # Parse system parameters for each system file
    system_params_list = [parse_system_parameters(system_file) for system_file in global_params.system_files]

    # Print mode information
    println("Running ML-IMC in the $(global_params.mode) mode.")

    return global_params, mc_params, nn_params, pretrain_params, system_params_list
end

function input_init(global_params::GlobalParameters, nn_params::NeuralNetParameters,
                    pretrain_params::PreTrainingParameters, system_params_list::Vector{SystemParameters})
    # Read reference data
    ref_rdfs = []
    for system_params in system_params_list
        bins, ref_rdf = read_rdf(system_params.rdf_file)
        append!(ref_rdfs, [ref_rdf])
    end

    # Initialize the model and the optimizer
    if global_params.model_file == "none"
        # Initialize the model
        println("Initializing a new neural network with random weights")

        model = model_init(nn_params)

        if global_params.optimizer_file != "none"
            println("Ignoring given optimizer filename...")
        end
        if global_params.gradients_file != "none"
            println("Ignoring given gradients filename...")
        end
        # Run pre-training if no initial model is given
        opt = init_optimizer(pretrain_params)
        # Restart the training
    else
        # Loading the model
        check_file(global_params.model_file)
        println("Reading model from $(global_params.model_file)")
        @load global_params.model_file model

        if global_params.mode == "training"
            # Either initialize the optimizer or read from a file
            if global_params.optimizer_file != "none"
                check_file(global_params.optimizer_file)
                println("Reading optimizer state from $(global_params.optimizer_file)")
                @load global_params.optimizer_file opt
            else
                opt = init_optimizer(nn_params)
            end

            # Optionally read gradients from a file
            if global_params.gradients_file != "none"
                check_file(global_params.gradients_file)
                println("Reading gradients from $(global_params.gradients_file)")
                @load global_params.gradients_file mean_loss_gradients
            end

            # Update the model if both opt and gradients are restored
            if global_params.optimizer_file != "none" && global_params.gradients_file != "none"
                println("\nUsing the restored gradients and optimizer to update the current model...\n")
                update_model!(model, opt, mean_loss_gradients)

                # Skip updating if no gradients are provided
            elseif global_params.optimizer_file != "none" && global_params.gradients_file == "none"
                println("\nNo gradients were provided, rerunning the training iteration with the current model and restored optimizer...\n")

                # Update the model if gradients are provided without the optimizer:
                # valid for optimizer that do not save their state, e.g. Descent,
                # otherwise might produce unexpected results
            elseif global_params.optimizer_file == "none" && global_params.gradients_file != "none"
                println("\nUsing the restored gradients with reinitialized optimizer to update the current model...\n")
                update_model!(model, opt, mean_loss_gradients)
            else
                println("\nNeither gradients nor optimizer were provided, rerunning the training iteration with the current model...\n")
            end
        end
    end

    if global_params.mode == "training"
        return (model, opt, ref_rdfs)
    else
        return (model)
    end
end

function init_optimizer(params::Union{NeuralNetParameters, PreTrainingParameters})
    function get_rate(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.learning_rate : params.learning_rate
    end

    function get_momentum(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.momentum : params.momentum
    end

    function get_decay1(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.decay_1 : params.decay_1
    end

    function get_decay2(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.decay_2 : params.decay_2
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

    optimizer_name = params isa NeuralNetParameters ? params.optimizer : params.optimizer
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
                                  nn_params::NeuralNetParameters)::Vector{Bool} where {T <: AbstractFloat}
    return distance_vector .< nn_params.max_distance_cutoff
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
        weights_shape = (system_params.n_bins, nn_params.neurons[layer_id - 1] * nn_params.neurons[layer_id])
        push!(cross_accumulators, zeros(weights_shape))

        if nn_params.bias
            bias_shape = (system_params.n_bins, nn_params.neurons[layer_id])
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
        descriptor_gradients[i] = -system_params.beta .* (cross_accumulators[i] - ensemble_correlations[i])
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
        reg_loss_gradient = @. 2 * nn_params.regularization * parameters
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
    if nn_params.regularization > zero(T)
        reg_loss = nn_params.regularization * sum(sum(abs2, p) for p in Flux.params(model))
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
                                    system_params_list::Vector{SystemParameters},
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

function collect_system_averages(outputs::Vector{MonteCarloAverages},
                                 reference_rdfs,
                                 system_params_list::Vector{SystemParameters},
                                 global_params::GlobalParameters,
                                 nn_params::NeuralNetParameters,
                                 model::Chain)::Tuple{Vector{MonteCarloAverages}, Vector{Float64}}
    total_loss::Float64 = 0.0
    system_outputs::Vector{MonteCarloAverages} = Vector{MonteCarloAverages}()
    system_losses::Vector{Float64} = Vector{Float64}()

    for (system_idx, system_params) in enumerate(system_params_list)
        println("   System $(system_params.system_name):")

        # Initialize collection vectors
        descriptors::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
        energies::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
        acceptance_ratios::Vector{Float64} = Vector{Float64}()
        max_displacements::Vector{Float64} = Vector{Float64}()

        # Training-specific accumulators
        cross_accumulators::Vector{Vector{Matrix{Float64}}} = Vector{Vector{Matrix{Float64}}}()
        symm_func_accumulators::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}()

        # Collect matching outputs
        for output in outputs
            if system_params.system_name == output.system_params.system_name
                push!(descriptors, output.descriptor)
                push!(energies, output.energies)
                push!(acceptance_ratios, output.acceptance_ratio)
                push!(max_displacements, output.step_size)

                if global_params.mode == "training"
                    push!(cross_accumulators, output.cross_accumulators)
                    push!(symm_func_accumulators, output.symmetry_matrix_accumulator)
                end
            end
        end

        # Compute averages
        avg_descriptor::Vector{Float64} = mean(descriptors)
        avg_energies::Vector{Float64} = mean(energies)
        avg_acceptance::Float64 = mean(acceptance_ratios)
        avg_displacement::Float64 = mean(max_displacements)

        # Training-specific averages
        avg_cross_acc::Union{Vector{Matrix{Float64}}, Nothing} = nothing
        avg_symm_func::Union{Matrix{Float64}, Nothing} = nothing

        if global_params.mode == "training"
            avg_cross_acc = mean(cross_accumulators)
            avg_symm_func = mean(symm_func_accumulators)
        end

        # Create system output
        system_output = MonteCarloAverages(avg_descriptor,
                                           avg_energies,
                                           avg_cross_acc,
                                           avg_symm_func,
                                           avg_acceptance,
                                           system_params,
                                           avg_displacement)

        # Print statistics
        println("       Acceptance ratio = ", round(avg_acceptance; digits=4))
        println("       Max displacement = ", round(avg_displacement; digits=4))

        # Compute and accumulate loss for training mode
        if global_params.mode == "training"
            system_loss::Float64 = compute_training_loss(system_output.descriptor,
                                                         reference_rdfs[system_idx],
                                                         model,
                                                         nn_params,
                                                         avg_displacement)
            total_loss += system_loss
            push!(system_losses, system_loss)
        end

        push!(system_outputs, system_output)
    end

    # Calculate and print average loss for training mode
    if global_params.mode == "training"
        total_loss /= length(system_params_list)
        println("   \nTotal Average Loss = ", round(total_loss; digits=8))
    end

    return (system_outputs, system_losses)
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

function train!(global_params::GlobalParameters,
                mc_params::MonteCarloParameters,
                nn_params::NeuralNetParameters,
                system_params_list::Vector{SystemParameters},
                model::Flux.Chain,
                optimizer::Flux.Optimise.AbstractOptimiser,
                ref_rdfs)
    for iteration in 1:(nn_params.iterations)
        iter_string = lpad(iteration, 2, "0")
        println("\nIteration $iteration...")

        # Monte Carlo sampling
        inputs = prepare_monte_carlo_inputs(global_params, mc_params, nn_params, system_params_list, model)
        outputs = pmap(mcsample!, inputs)

        # Process system outputs and compute losses
        system_outputs, system_losses = collect_system_averages(outputs, ref_rdfs, system_params_list, global_params,
                                                                nn_params, model)

        # Compute gradients for each system
        loss_gradients = Vector{Any}(undef, length(system_outputs))
        for (system_id, system_output) in enumerate(system_outputs)
            system_params = system_params_list[system_id]

            loss_gradients[system_id] = compute_loss_gradients(system_output.cross_accumulators,
                                                               system_output.symmetry_matrix_accumulator,
                                                               system_output.descriptor,
                                                               ref_rdfs[system_id],
                                                               model,
                                                               system_params,
                                                               nn_params)

            # Save system outputs
            name = system_params.system_name
            write_rdf("RDFNN-$(name)-iter-$(iter_string).dat", system_output.descriptor, system_params)
            write_energies("energies-$(name)-iter-$(iter_string).dat",
                           system_output.energies,
                           mc_params,
                           system_params,
                           1)
        end

        # Compute mean loss gradients
        mean_loss_gradients = if global_params.adaptive_scaling
            gradient_coeffs = compute_adaptive_gradient_coefficients(system_losses)

            println("\nGradient scaling:")
            for (coeff, params) in zip(gradient_coeffs, system_params_list)
                println("   System $(params.system_name): $(round(coeff; digits=8))")
            end

            sum(loss_gradients .* gradient_coeffs)
        else
            mean(loss_gradients)
        end

        # Save model state
        for (filename, data) in [
            ("model-iter-$(iter_string).bson", model),
            ("opt-iter-$(iter_string).bson", optimizer),
            ("gradients-iter-$(iter_string).bson", mean_loss_gradients)
        ]
            @save filename data
            check_file(filename)
        end

        # Update model with computed gradients
        update_model!(model, optimizer, mean_loss_gradients)
    end

    println("Training completed!")
end

# -----------------------------------------------------------------------------
# --- Simulation

function simulate!(model::Flux.Chain,
                   global_params::GlobalParameters,
                   mc_params::MonteCarloParameters,
                   nn_params::NeuralNetParameters,
                   system_params::SystemParameters)

    # Create input for each worker
    input = MonteCarloSampleInput(global_params, mc_params, nn_params, system_params, model)
    inputs = fill(input, nworkers())

    # Run parallel Monte Carlo sampling
    outputs = pmap(mcsample!, inputs)

    # Collect system statistics
    system_outputs, system_losses = collect_system_averages(outputs,
                                                            nothing,
                                                            [system_params],
                                                            global_params,
                                                            nothing,
                                                            nothing)

    # Save simulation results
    system_name = system_params.system_name
    try
        write_rdf("RDFNN-$(system_name).dat", system_outputs[1].descriptor, system_params)
        write_energies("energies-$(system_name).dat", system_outputs[1].energies, mc_params, system_params, 1)
    catch e
        @error "Failed to save simulation results" exception=e system=system_name
        rethrow()
    end
end

# -----------------------------------------------------------------------------
# --- Symmetry Function

function combine_symmetry_matrices(g2_matrix, g3_matrix, g9_matrix) # NOTE: better no types
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
    n_g2_functions = length(nn_params.g2_functions)
    g2_matrix = Matrix{T}(undef, n_atoms, n_g2_functions)

    for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]
        for (j, g2_func) in enumerate(nn_params.g2_functions)
            g2_matrix[i, j] = compute_g2(distance_vector,
                                         g2_func.eta,
                                         g2_func.rcutoff,
                                         g2_func.rshift)
        end
    end

    return nn_params.symm_function_scaling == one(T) ? g2_matrix :
           g2_matrix .* nn_params.symm_function_scaling
end

function update_g2_matrix!(g2_matrix::AbstractMatrix{T},
                           distance_vector1::AbstractVector{T},
                           distance_vector2::AbstractVector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           point_index::Integer)::AbstractMatrix{T} where {T <: AbstractFloat}
    scaling = nn_params.symm_function_scaling

    # Update displaced particle row
    @inbounds for (j, g2_func) in enumerate(nn_params.g2_functions)
        g2_matrix[point_index, j] = compute_g2(distance_vector2,
                                               g2_func.eta,
                                               g2_func.rcutoff,
                                               g2_func.rshift) * scaling
    end

    # Update affected atoms
    @inbounds for i in 1:(system_params.n_atoms)
        i == point_index && continue

        for (j, g2_func) in enumerate(nn_params.g2_functions)
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
    n_g3_functions = length(nn_params.g3_functions)
    g3_matrix = Matrix{T}(undef, n_atoms, n_g3_functions)

    @inbounds for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]

        for (j, g3_func) in enumerate(nn_params.g3_functions)
            g3_matrix[i, j] = compute_g3(i, coordinates, box, distance_vector,
                                         g3_func.rcutoff, g3_func.eta,
                                         g3_func.zeta, g3_func.lambda,
                                         g3_func.rshift)
        end
    end

    scaling = nn_params.symm_function_scaling
    return scaling == one(T) ? g3_matrix : rmul!(g3_matrix, scaling)
end

function update_g3_matrix!(g3_matrix::Matrix{T},
                           coordinates1::Matrix{T},
                           coordinates2::Matrix{T},
                           box::Vector{T},
                           distance_vec_1::Vector{T},
                           distance_vec_2::Vector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           displaced_atom_index::Integer)::Matrix{T} where {T <: AbstractFloat}
    scaling_factor::T = nn_params.symm_function_scaling
    n_atoms::Int = system_params.n_atoms

    @inbounds for central_atom_idx in 1:n_atoms
        if central_atom_idx == displaced_atom_index
            # Update matrix for displaced atom
            @inbounds for (g3_idx, g3_func) in enumerate(nn_params.g3_functions)
                g3_matrix[central_atom_idx, g3_idx] = compute_g3(displaced_atom_index,
                                                                 coordinates2,
                                                                 box,
                                                                 distance_vec_2,
                                                                 g3_func.rcutoff,
                                                                 g3_func.eta,
                                                                 g3_func.zeta,
                                                                 g3_func.lambda,
                                                                 g3_func.rshift) * scaling_factor
            end
        else
            # Update matrix for other atoms
            @inbounds for (g3_idx, g3_func) in enumerate(nn_params.g3_functions)
                r_cutoff::T = g3_func.rcutoff
                dist_ij_1::T = distance_vec_1[central_atom_idx]
                dist_ij_2::T = distance_vec_2[central_atom_idx]

                # Check if atom is within cutoff distance
                if (zero(T) < dist_ij_1 < r_cutoff) || (zero(T) < dist_ij_2 < r_cutoff)
                    central_atom_pos::Vector{T} = @view coordinates2[:, central_atom_idx]
                    delta_g3::T = zero(T)

                    # Calculate G3 changes for all third atoms
                    @inbounds for third_atom_idx in 1:n_atoms
                        # Skip if atoms are identical
                        if third_atom_idx == displaced_atom_index ||
                           third_atom_idx == central_atom_idx
                            continue
                        end

                        third_atom_pos::Vector{T} = @view coordinates2[:, third_atom_idx]
                        dist_ik::T = compute_distance(central_atom_pos, third_atom_pos, box)

                        # Check if third atom is within cutoff
                        if zero(T) < dist_ik < r_cutoff
                            displaced_pos_1::Vector{T} = @view coordinates1[:, displaced_atom_index]
                            displaced_pos_2::Vector{T} = @view coordinates2[:, displaced_atom_index]

                            dist_kj_1::T = compute_distance(displaced_pos_1, third_atom_pos, box)
                            dist_kj_2::T = compute_distance(displaced_pos_2, third_atom_pos, box)

                            if (zero(T) < dist_kj_1 < r_cutoff) ||
                               (zero(T) < dist_kj_2 < r_cutoff)
                                # Calculate angle vectors
                                vec_ij_1::Vector{T} = compute_directional_vector(central_atom_pos, displaced_pos_1, box)
                                vec_ij_2::Vector{T} = compute_directional_vector(central_atom_pos, displaced_pos_2, box)
                                vec_ik::Vector{T} = compute_directional_vector(central_atom_pos, third_atom_pos, box)

                                # Calculate cosine angles
                                cos_angle_1::T = dot(vec_ij_1, vec_ik) / (dist_ij_1 * dist_ik)
                                cos_angle_2::T = dot(vec_ij_2, vec_ik) / (dist_ij_2 * dist_ik)

                                # Ensure angles are valid
                                @assert -one(T) <= cos_angle_1 <= one(T)
                                @assert -one(T) <= cos_angle_2 <= one(T)

                                # Calculate G3 differences
                                g3_val_1::T = compute_g3_element(cos_angle_1, dist_ij_1, dist_ik, dist_kj_1,
                                                                 r_cutoff, g3_func.eta, g3_func.zeta,
                                                                 g3_func.lambda, g3_func.rshift)
                                g3_val_2::T = compute_g3_element(cos_angle_2, dist_ij_2, dist_ik, dist_kj_2,
                                                                 r_cutoff, g3_func.eta, g3_func.zeta,
                                                                 g3_func.lambda, g3_func.rshift)

                                delta_g3 += T(2)^(one(T) - g3_func.zeta) * (g3_val_2 - g3_val_1)
                            end
                        end
                    end

                    # Update matrix element with accumulated changes
                    g3_matrix[central_atom_idx, g3_idx] += delta_g3 * scaling_factor
                end
            end
        end
    end

    return g3_matrix
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

function compute_g9(atom_index::Integer,
                    coordinates::Matrix{T},
                    box::Vector{T},
                    distance_vector::Vector{T},
                    r_cutoff::T,
                    η::T,
                    ζ::T,
                    λ::T,
                    r_shift::T)::T where {T <: AbstractFloat}
    accumulator = zero(T)

    @inbounds for k in eachindex(distance_vector)
        distance_ik = distance_vector[k]

        @inbounds @simd for j in 1:(k - 1)
            distance_ij = distance_vector[j]

            if zero(T) < distance_ij < r_cutoff && zero(T) < distance_ik < r_cutoff
                cos_angle = compute_cos_angle(coordinates,
                                              box,
                                              atom_index,
                                              j,
                                              k,
                                              distance_ij,
                                              distance_ik)

                accumulator += compute_g9_element(cos_angle,
                                                  distance_ij,
                                                  distance_ik,
                                                  r_cutoff,
                                                  η,
                                                  ζ,
                                                  λ,
                                                  r_shift)
            end
        end
    end

    return (T(2)^(one(T) - ζ) * accumulator)
end

function build_g9_matrix(distance_matrix::AbstractMatrix{T},
                         coordinates::AbstractMatrix{T},
                         box::AbstractVector{T},
                         nn_params::NeuralNetParameters)::Matrix{T} where {T <: AbstractFloat}
    n_atoms = size(distance_matrix, 1)
    n_g9_functions = length(nn_params.g9_functions)
    g9_matrix = Matrix{T}(undef, n_atoms, n_g9_functions)

    @inbounds for i in 1:n_atoms
        distance_vector = @view distance_matrix[i, :]

        for (j, g9_func) in enumerate(nn_params.g9_functions)
            g9_matrix[i, j] = compute_g9(i, coordinates, box, distance_vector,
                                         g9_func.rcutoff, g9_func.eta,
                                         g9_func.zeta, g9_func.lambda,
                                         g9_func.rshift)
        end
    end

    scaling = nn_params.symm_function_scaling
    return scaling == one(T) ? g9_matrix : rmul!(g9_matrix, scaling)
end

function update_g9_matrix!(g9_matrix::AbstractMatrix{T},
                           coordinates1::AbstractMatrix{T},
                           coordinates2::AbstractMatrix{T},
                           box::AbstractVector{T},
                           distance_vector1::AbstractVector{T},
                           distance_vector2::AbstractVector{T},
                           system_params::SystemParameters,
                           nn_params::NeuralNetParameters,
                           displaced_atom_index::Integer) where {T <: AbstractFloat}
    for selected_atom_index in 1:(system_params.n_atoms)
        if selected_atom_index == displaced_atom_index
            for (g9_index, g9_func) in enumerate(nn_params.g9_functions)
                g9_matrix[selected_atom_index, g9_index] = compute_g9(displaced_atom_index, coordinates2, box,
                                                                      distance_vector2,
                                                                      g9_func.rcutoff, g9_func.eta, g9_func.zeta,
                                                                      g9_func.lambda, g9_func.rshift) *
                                                           nn_params.symm_function_scaling
            end
        else
            for (g9_index, g9_func) in enumerate(nn_params.g9_functions)
                distance_ij_1 = distance_vector1[selected_atom_index]
                distance_ij_2 = distance_vector2[selected_atom_index]

                if 0 < distance_ij_2 < g9_func.rcutoff || 0 < distance_ij_1 < g9_func.rcutoff
                    Δg9 = zero(T)

                    for third_atom_index in 1:(system_params.n_atoms)
                        if third_atom_index != displaced_atom_index && third_atom_index != selected_atom_index
                            selected_atom = @view coordinates2[:, selected_atom_index]
                            third_atom = @view coordinates2[:, third_atom_index]
                            distance_ik = compute_distance(selected_atom, third_atom, box)

                            if 0 < distance_ik < g9_func.rcutoff
                                displaced_atom_1 = @view coordinates1[:, displaced_atom_index]
                                displaced_atom_2 = @view coordinates2[:, displaced_atom_index]

                                vector_ij_1 = compute_directional_vector(selected_atom, displaced_atom_1, box)
                                vector_ij_2 = compute_directional_vector(selected_atom, displaced_atom_2, box)
                                vector_ik = compute_directional_vector(selected_atom, third_atom, box)

                                cos_angle1 = dot(vector_ij_1, vector_ik) / (distance_ij_1 * distance_ik)
                                cos_angle2 = dot(vector_ij_2, vector_ik) / (distance_ij_2 * distance_ik)

                                @assert -1≤cos_angle1≤1 "Invalid cosine value: $cos_angle1"
                                @assert -1≤cos_angle2≤1 "Invalid cosine value: $cos_angle2"

                                g9_1 = compute_g9_element(cos_angle1, distance_ij_1, distance_ik,
                                                          g9_func.rcutoff, g9_func.eta, g9_func.zeta,
                                                          g9_func.lambda, g9_func.rshift)
                                g9_2 = compute_g9_element(cos_angle2, distance_ij_2, distance_ik,
                                                          g9_func.rcutoff, g9_func.eta, g9_func.zeta,
                                                          g9_func.lambda, g9_func.rshift)

                                Δg9 += 2^(1 - g9_func.zeta) * (g9_2 - g9_1)
                            end
                        end
                    end

                    g9_matrix[selected_atom_index, g9_index] += Δg9 * nn_params.symm_function_scaling
                end
            end
        end
    end

    return g9_matrix
end

# -----------------------------------------------------------------------------
# --- Monte Carlo

function update_distance_histogram!(distance_matrix::Matrix{T},
                                    histogram::Vector{T},
                                    system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_atoms = system_params.n_atoms
    bin_width = system_params.bin_width
    n_bins = system_params.n_bins

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

function update_distance_histogram_vectors!(histogram::Vector{T},
                                            old_distances::Vector{T},
                                            new_distances::Vector{T},
                                            system_params::SystemParameters)::Vector{T} where {T <: AbstractFloat}
    n_atoms = system_params.n_atoms
    bin_width = system_params.bin_width
    n_bins = system_params.n_bins

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
    n_pairs = system_params.n_atoms * (system_params.n_atoms - 1) ÷ 2

    # Pre-calculate bin radii and shell volumes
    bin_width = system_params.bin_width
    bins = [(i * bin_width) for i in 1:(system_params.n_bins)]
    shell_volumes = [4 * π * system_params.bin_width * bins[i]^2 for i in eachindex(bins)]

    # Calculate normalization factors
    normalization_factors = fill(box_volume / n_pairs, system_params.n_bins) ./ shell_volumes

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

function mcmove!(mc_arrays::Tuple{Frame, Matrix{T}, Matrix{T}, Union{Matrix{T}, Vector{T}},
                                  Union{Matrix{T}, Vector{T}}},
                 current_energy::T,
                 energy_prev_vec::Vector{T},
                 model::Flux.Chain,
                 nn_params::NeuralNetParameters,
                 system_params::SystemParameters,
                 box::Vector{T},
                 rng::Xoroshiro128Plus,
                 step_size::T)::Tuple{Tuple{Frame, Matrix{T}, Matrix{T}, Union{Matrix{T}, Vector{T}},
                                            Union{Matrix{T}, Vector{T}}},
                                      T,
                                      Vector{T},
                                      Int} where {T <: AbstractFloat}

    # Unpack simulation arrays
    frame, distance_matrix, g2_matrix, g3_matrix, g9_matrix = mc_arrays

    # Store original coordinates if needed for angular terms
    coordinates_orig = (!isempty(g3_matrix) || !isempty(g9_matrix)) ? copy(positions(frame)) : nothing

    # Select random particle and get its initial distances
    particle_index = rand(rng, 1:(system_params.n_atoms))
    distances_orig = @view distance_matrix[:, particle_index]
    energy_orig = copy(current_energy)

    # Displace particle
    displacement = step_size * (rand(rng, T, 3) .- 0.5)
    positions(frame)[:, particle_index] .+= displacement
    apply_periodic_boundaries!(frame, box, particle_index)

    # Compute new distances
    new_position = positions(frame)[:, particle_index]
    distances_new = compute_distance_vector(new_position, positions(frame), box)

    # Update symmetry matrices
    indexes_for_update = get_energies_update_mask(distances_new, nn_params)

    g2_matrix_new = copy(g2_matrix)
    update_g2_matrix!(g2_matrix_new, distances_orig, distances_new, system_params, nn_params, particle_index)

    g3_matrix_new = copy(g3_matrix)
    if !isempty(g3_matrix)
        update_g3_matrix!(g3_matrix_new, coordinates_orig, positions(frame), box,
                          distances_orig, distances_new, system_params, nn_params, particle_index)
    end

    g9_matrix_new = copy(g9_matrix)
    if !isempty(g9_matrix)
        update_g9_matrix!(g9_matrix_new, coordinates_orig, positions(frame), box,
                          distances_orig, distances_new, system_params, nn_params, particle_index)
    end

    # Compute new energy
    symmetry_matrix = combine_symmetry_matrices(g2_matrix_new, g3_matrix_new, g9_matrix_new)
    energy_vector_new = update_system_energies_vector(symmetry_matrix, model, indexes_for_update, energy_prev_vec)
    energy_new = sum(energy_vector_new)

    # Accept/reject move using Metropolis criterion
    accepted = 0
    if rand(rng, T) < exp(-((energy_new - energy_orig) * system_params.beta))
        accepted = 1
        current_energy = energy_new
        distance_matrix[particle_index, :] = distances_new
        distance_matrix[:, particle_index] = distances_new
        return ((frame, distance_matrix, g2_matrix_new, g3_matrix_new, g9_matrix_new),
                current_energy, energy_vector_new, accepted)
    else
        positions(frame)[:, particle_index] .-= displacement
        apply_periodic_boundaries!(frame, box, particle_index)
        return ((frame, distance_matrix, g2_matrix, g3_matrix, g9_matrix),
                current_energy, energy_prev_vec, accepted)
    end
end

function mcsample!(input::MonteCarloSampleInput)::MonteCarloAverages
    # Unpack input parameters with improved naming
    model = input.model
    global_params = input.global_params
    mc_params = input.mc_params
    nn_params = input.nn_params
    system_params = input.system_params

    # Initialize simulation parameters
    step_size::Float64 = copy(system_params.max_displacement)

    # Set worker ID and output files
    worker_id::Int = nprocs() == 1 ? myid() : myid() - 1
    worker_id_str::String = lpad(worker_id, 3, "0")

    traj_file::String = "mctraj-p$(worker_id_str).xtc"
    pdb_file::String = "confin-p$(worker_id_str).pdb"

    # Initialize RNG
    rng::Xoroshiro128Plus = Xoroshiro128Plus()

    # Initialize frame based on mode
    frame::Frame = if global_params.mode == "training"
        trajectory = read_xtc(system_params)
        n_frames::Int = Int(size(trajectory)) - 1
        frame_id::Int = rand(rng, 1:n_frames)
        deepcopy(read_step(trajectory, frame_id))
    else
        pdb = read_pdb(system_params)
        deepcopy(read_step(pdb, 0))
    end

    # Get simulation box parameters
    box::Vector{Float64} = lengths(UnitCell(frame))

    # Initialize trajectory output if verbose mode
    if global_params.output_mode == "verbose"
        write_trajectory(positions(frame), box, system_params, traj_file, 'w')
        write_trajectory(positions(frame), box, system_params, pdb_file, 'w')
    end

    # Calculate data collection parameters
    total_points::Int = Int(mc_params.steps / mc_params.output_frequency)
    production_points::Int = Int((mc_params.steps - mc_params.equilibration_steps) / mc_params.output_frequency)

    # Initialize distance and symmetry matrices
    distance_matrix::Matrix{Float64} = build_distance_matrix(frame)
    g2_matrix::Matrix{Float64} = build_g2_matrix(distance_matrix, nn_params)

    g3_matrix::Union{Matrix{Float64}, Vector{Float64}} = if !isempty(nn_params.g3_functions)
        build_g3_matrix(distance_matrix, positions(frame), box, nn_params)
    else
        Float64[]
    end

    g9_matrix::Union{Matrix{Float64}, Vector{Float64}} = if !isempty(nn_params.g9_functions)
        build_g9_matrix(distance_matrix, positions(frame), box, nn_params)
    else
        Float64[]
    end

    # Create MC arrays tuple
    mc_arrays::Tuple = (frame, distance_matrix, g2_matrix, g3_matrix, g9_matrix)

    # Initialize histogram accumulator
    hist_accumulator::Vector{Float64} = zeros(Float64, system_params.n_bins)

    # Initialize training-specific arrays if in training mode
    if global_params.mode == "training"
        hist::Vector{Float64} = zeros(Float64, system_params.n_bins)
        g2_accumulator::Matrix{Float64} = zeros(size(g2_matrix))
        g3_accumulator::Union{Matrix{Float64}, Vector{Float64}} = zeros(size(g3_matrix))
        g9_accumulator::Union{Matrix{Float64}, Vector{Float64}} = zeros(size(g9_matrix))
        cross_accumulators::Vector{Matrix{Float64}} = initialize_cross_accumulators(nn_params, system_params)
    end

    # Initialize energy calculations
    symm_func_matrix::Matrix{Float64} = combine_symmetry_matrices(g2_matrix, g3_matrix, g9_matrix)
    energy_vector::Vector{Float64} = init_system_energies_vector(symm_func_matrix, model)
    total_energy::Float64 = sum(energy_vector)
    energies::Vector{Float64} = zeros(total_points + 1)
    energies[1] = total_energy

    # Initialize acceptance counters
    accepted_total::Int = 0
    accepted_intermediate::Int = 0

    # Main Monte Carlo loop
    @fastmath for step in 1:(mc_params.steps)
        mc_arrays, total_energy, energy_vector, accepted = mcmove!(mc_arrays, total_energy, energy_vector, model,
                                                                   nn_params, system_params, box, rng, step_size)
        accepted_total += accepted
        accepted_intermediate += accepted

        # Adjust step size during equilibration
        if mc_params.step_adjust_frequency > 0 &&
           step % mc_params.step_adjust_frequency == 0 &&
           step < mc_params.equilibration_steps
            step_size = adjust_monte_carlo_step!(step_size, system_params, box, mc_params, accepted_intermediate)
            accepted_intermediate = 0
        end

        # Store energy data
        if step % mc_params.output_frequency == 0
            energies[Int(step / mc_params.output_frequency) + 1] = total_energy
        end

        # Write trajectory if in verbose mode
        if global_params.output_mode == "verbose" &&
           step % mc_params.trajectory_output_frequency == 0
            write_trajectory(positions(mc_arrays[1]), box, system_params, traj_file, 'a')
        end

        # Update histograms and accumulators in production phase
        if step % mc_params.output_frequency == 0 && step > mc_params.equilibration_steps
            frame, distance_matrix, g2_matrix, g3_matrix, g9_matrix = mc_arrays

            if global_params.mode == "training"
                hist = update_distance_histogram!(distance_matrix, hist, system_params)
                hist_accumulator .+= hist
                g2_accumulator .+= g2_matrix

                if !isempty(g3_matrix)
                    g3_accumulator .+= g3_matrix
                end
                if !isempty(g9_matrix)
                    g9_accumulator .+= g9_matrix
                end

                normalize_hist_to_rdf!(hist, system_params, box)
                symm_func_matrix = combine_symmetry_matrices(g2_matrix, g3_matrix, g9_matrix)
                update_cross_accumulators!(cross_accumulators, symm_func_matrix, hist, model, nn_params)
                hist = zeros(Float64, system_params.n_bins)
            else
                hist_accumulator = update_distance_histogram!(distance_matrix, hist_accumulator, system_params)
            end
        end
    end

    # Calculate final acceptance ratio
    acceptance_ratio::Float64 = accepted_total / mc_params.steps

    # Process final results
    if global_params.mode == "training"
        # Normalize accumulators
        for cross in cross_accumulators
            cross ./= production_points
        end

        g2_accumulator ./= production_points
        if !isempty(g3_matrix)
            g3_accumulator ./= production_points
        end
        if !isempty(g9_matrix)
            g9_accumulator ./= production_points
        end

        symm_func_accumulator = combine_symmetry_matrices(g2_accumulator, g3_accumulator, g9_accumulator)
    end

    # Normalize final histogram
    hist_accumulator ./= production_points
    normalize_hist_to_rdf!(hist_accumulator, system_params, box)

    # Return results based on mode
    if global_params.mode == "training"
        return MonteCarloAverages(hist_accumulator,
                                  energies,
                                  cross_accumulators,
                                  symm_func_accumulator,
                                  acceptance_ratio,
                                  system_params,
                                  step_size)
    else
        return MonteCarloAverages(hist_accumulator,
                                  energies,
                                  nothing,
                                  nothing,
                                  acceptance_ratio,
                                  system_params,
                                  step_size)
    end
end

function adjust_monte_carlo_step!(current_step_size::T,
                                  system_params::SystemParameters,
                                  box::AbstractVector{T},
                                  mc_params::MonteCarloParameters,
                                  accepted_moves::Integer)::T where {T <: AbstractFloat}
    # Calculate current acceptance ratio
    acceptance_ratio = accepted_moves / mc_params.step_adjust_frequency

    # Adjust step size based on target acceptance ratio
    current_step_size = (acceptance_ratio / system_params.target_acceptance_ratio) * current_step_size

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
    n_bins = system_params.n_bins
    β = system_params.beta

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
    nn_params = input.nn_params
    system_params = input.system_params
    ref_rdf = input.reference_rdf

    println("Pre-computing data for $(system_params.system_name)...")

    # Compute potential of mean force
    pmf = compute_pmf(ref_rdf, system_params)

    # Initialize trajectory reading
    trajectory = read_xtc(system_params)
    n_frames = Int(size(trajectory)) - 1  # Skip first frame

    # Pre-allocate data containers
    distance_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    histograms = Vector{Vector{Float64}}(undef, n_frames)
    g2_matrices = Vector{Matrix{Float64}}(undef, n_frames)
    g3_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.g3_functions) > 0))
    g9_matrices = Vector{Matrix{Float64}}(undef, n_frames * (length(nn_params.g9_functions) > 0))

    # Process each frame
    for frame_id in 1:n_frames
        frame = read_step(trajectory, frame_id)
        box = lengths(UnitCell(frame))
        coords = positions(frame)

        # Compute distance matrix and histogram
        distance_matrices[frame_id] = build_distance_matrix(frame)
        histograms[frame_id] = zeros(Float64, system_params.n_bins)
        update_distance_histogram!(distance_matrices[frame_id], histograms[frame_id], system_params)

        # Compute symmetry function matrices
        g2_matrices[frame_id] = build_g2_matrix(distance_matrices[frame_id], nn_params)

        if !isempty(nn_params.g3_functions)
            g3_matrices[frame_id] = build_g3_matrix(distance_matrices[frame_id], coords, box, nn_params)
        end

        if !isempty(nn_params.g9_functions)
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
    reg_loss = pretrain_params.regularization * sum(abs2, model_params[1])

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
    reg_gradient = @. model_params * 2 * pretrain_params.regularization

    return loss_gradient + reg_gradient
end

function pretraining_move!(reference_data::ReferenceData, model::Flux.Chain, nn_params::NeuralNetParameters,
                           system_params::SystemParameters, rng::Xoroshiro128Plus)
    # Pick a frame
    traj = read_xtc(system_params)
    nframes = Int(size(traj)) - 1 # Don't take the first frame
    frame_id::Int = rand(rng, 1:nframes)
    frame = read_step(traj, frame_id)
    box = lengths(UnitCell(frame))
    # Pick a particle
    point_index::Int = rand(rng, 1:(system_params.n_atoms))

    # Read reference data
    distance_matrix = reference_data.distance_matrices[frame_id]
    hist = reference_data.histograms[frame_id]
    pmf = reference_data.pmf

    # If no angular symmetry functions are provided, use G2 only
    if isempty(reference_data.g3_matrices) && isempty(reference_data.g9_matrices)
        symm_func_matrix1 = reference_data.g2_matrices[frame_id]
    else
        # Make a copy of the original coordinates
        coordinates1 = copy(positions(frame))
        # Combine symmetry function matrices
        if isempty(reference_data.g3_matrices)
            symm_func_matrices = [reference_data.g2_matrices[frame_id], reference_data.g9_matrices[frame_id]]
        elseif isempty(reference_data.g9_matrices)
            symm_func_matrices = [reference_data.g2_matrices[frame_id], reference_data.g3_matrices[frame_id]]
        else
            symm_func_matrices = [
                reference_data.g2_matrices[frame_id],
                reference_data.g3_matrices[frame_id],
                reference_data.g9_matrices[frame_id]
            ]
        end
        # Unpack symmetry functions and concatenate horizontally into a single matrix
        symm_func_matrix1 = hcat(symm_func_matrices...)
    end

    # Compute energy of the initial configuration
    e_nn1_vector = init_system_energies_vector(symm_func_matrix1, model)
    e_nn1 = sum(e_nn1_vector)
    e_pmf1 = sum(hist .* pmf)

    # Allocate the distance vector
    distance_vector1 = distance_matrix[:, point_index]

    # Displace the particle
    dr = system_params.max_displacement * (rand(rng, Float64, 3) .- 0.5)

    positions(frame)[:, point_index] .+= dr

    # Compute the updated distance vector
    point = positions(frame)[:, point_index]
    distance_vector2 = compute_distance_vector(point, positions(frame), box)

    # Update the histogram
    hist = update_distance_histogram_vectors!(hist, distance_vector1, distance_vector2, system_params)

    # Get indexes for updating ENN
    indexes_for_update = get_energies_update_mask(distance_vector2, nn_params)

    # Make a copy of the original G2 matrix and update it
    g2_matrix2 = copy(reference_data.g2_matrices[frame_id])
    update_g2_matrix!(g2_matrix2, distance_vector1, distance_vector2, system_params, nn_params, point_index)

    # Make a copy of the original angular matrices and update them
    g3_matrix2 = []
    if !isempty(reference_data.g3_matrices)
        g3_matrix2 = copy(reference_data.g3_matrices[frame_id])
        update_g3_matrix!(g3_matrix2, coordinates1, positions(frame), box, distance_vector1, distance_vector2,
                          system_params,
                          nn_params, point_index)
    end

    # Make a copy of the original angular matrices and update them
    g9_matrix2 = []
    if !isempty(reference_data.g9_matrices)
        g9_matrix2 = copy(reference_data.g9_matrices[frame_id])
        update_g9_matrix!(g9_matrix2, coordinates1, positions(frame), box, distance_vector1, distance_vector2,
                          system_params,
                          nn_params, point_index)
    end

    # Combine symmetry function matrices accumulators
    symm_func_matrix2 = combine_symmetry_matrices(g2_matrix2, g3_matrix2, g9_matrix2)

    # Compute the NN energy again
    e_nn2_vector = update_system_energies_vector(symm_func_matrix2, model, indexes_for_update, e_nn1_vector)
    e_nn2 = sum(e_nn2_vector)
    e_pmf2 = sum(hist .* pmf)

    # Get the energy differences
    Δe_nn = e_nn2 - e_nn1
    Δe_pmf = e_pmf2 - e_pmf1

    # Revert the changes in the frame arrays
    positions(frame)[:, point_index] .-= dr
    hist = update_distance_histogram_vectors!(hist, distance_vector2, distance_vector1, system_params)

    return (symm_func_matrix1, symm_func_matrix2, Δe_nn, Δe_pmf)
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
        println("\nStarting pre-training with Monte Carlo for $(pretrain_params.regularization) steps")
        println("Regularization parameter: $(pretrain_params.regularization)")
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
    for step in 1:(pretrain_params.steps)
        should_report = (step % pretrain_params.output_frequency == 0) || (step == 1)
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
        if global_params.model_file == "none"
            model = pretrain_model!(pretrain_params, nn_params, system_params_list, model, optimizer, ref_rdfs)

            println("\nRe-initializing the optimizer for the training...")
            optimizer = init_optimizer(nn_params)
            report_optimizer(optimizer)
            println("Neural network regularization parameter: $(nn_params.regularization)")
        end

        # Execute main training
        println("""
            \nStarting main training phase:
            - Adaptive gradient scaling: $(global_params.adaptive_scaling)
            - Iterations: $(nn_params.iterations)
            - Running on $(num_workers) worker(s)
            - Total steps: $(mc_params.steps * num_workers / 1e6)M
            - Equilibration steps per rank: $(mc_params.equilibration_steps / 1e6)M
            """)

        train!(global_params, mc_params, nn_params, system_params_list, model, optimizer, ref_rdfs)
    else
        length(system_params_list) == 1 || throw(ArgumentError("Simulation mode supports only one system"))
        println("Running simulation with $(global_params.model_file)")
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
