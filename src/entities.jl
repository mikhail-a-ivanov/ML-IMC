using ..ML_IMC

struct G2
    eta::Float64
    rcutoff::Float64
    rshift::Float64
    norm::Float64
end

struct LRSchedulerConfig
    warmup_epochs::Int
    warmup_start_lr::Float64
    patience::Int
    factor::Float64
    min_lr::Float64
    cooldown::Int
end

mutable struct LRSchedulerState
    current_lr::Float64
    best_loss::Float64
    bad_epochs::Int
    cooldown_left::Int
end

struct NeuralNetParameters
    # Symmetry functions
    g2_functions::Vector{G2}
    max_distance_cutoff::Float64
    symm_function_scaling::Float64

    # Architecture
    neurons::Vector{Int}
    activations::Vector{String}
    bias::Bool

    # Training parameters
    iterations::Int
    regularization::Float64

    # Gradient settings
    gradient_type::String

    # Optimizer settings
    optimizer::String
    learning_rate::Float64
    momentum::Float64
    decay_1::Float64
    decay_2::Float64

    # LR scheduler
    lr_scheduler_config::LRSchedulerConfig
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
    output_dir::String
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
    use_diff_gradient::Bool
    use_all_particles::Bool
    gradient_type::String
    save_frequency::Int
    output_prefix::String
    output_dir::String

    # LR scheduler
    lr_scheduler_config::LRSchedulerConfig
end

struct MagicPreTrainingParameters
    model_file::String
    potential_files::Vector{String}
    use_diff_gradient::Bool
    use_all_particles::Bool
end
