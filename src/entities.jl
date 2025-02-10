using ..ML_IMC

struct G2
    eta::Float64
    rcutoff::Float64
    rshift::Float64
    norm::Float64
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
    batch_size::Int
    output_frequency::Int
    regularization::Float64
    optimizer::String
    learning_rate::Float64
    momentum::Float64
    decay_1::Float64
    decay_2::Float64
end
