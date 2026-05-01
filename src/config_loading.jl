using ..ML_IMC

function parse_symmetry_functions(filename::String)
    symm_data = TOML.parsefile(filename)

    println("----------------------------- Symmetry Functions ------------------------------")
    println("Symmetry Functions file: $(filename)")
    println()
    TOML.print(symm_data)
    println()

    g2_functions = Vector{G2}()
    scaling_factor = Float32(get(symm_data, "scaling", 1.0f0))
    max_cutoff = 0.0f0

    if haskey(symm_data, "G2")
        for params in symm_data["G2"]
            eta, rcutoff, rshift, norm = params
            g2 = G2(Float32(eta), Float32(rcutoff), Float32(rshift), Float32(norm))
            push!(g2_functions, g2)
            max_cutoff = max(max_cutoff, Float32(rcutoff))
        end
    end

    return g2_functions, max_cutoff, scaling_factor
end

function parse_system_parameters(filename::String)
    # Constants
    NA = 6.0221406f23  # Avogadro constant
    kB = 1.3806485f-23 * NA / 1000.0f0  # Boltzmann constant in kJ/(mol·K)

    system_data = TOML.parsefile(filename)["system"]

    # Read and validate topology file
    pdb = Trajectory(system_data["topology_file_path"])
    frame = read(pdb)
    n_atoms = length(frame)
    atom_name = name(Atom(frame, 1))

    # Read RDF data
    bins, rdf = read_rdf(system_data["rdf_file_path"])
    n_bins = length(bins)
    bin_width = infer_bin_width(bins)

    # Calculate beta from temperature
    temperature = Float32(system_data["temperature"])
    beta = 1.0f0 / (kB * temperature)

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
                            Float32(system_data["max_displacement"]),
                            Float32(system_data["target_acceptance_ratio"]))
end

function parse_lr_scheduler(section)
    return LRSchedulerConfig(get(section, "warmup_epochs", 0),
                             Float32(get(section, "warmup_start", 1.0f-7)),
                             get(section, "patience", 50),
                             Float32(get(section, "factor", 0.5f0)),
                             Float32(get(section, "min_lr", 1.0f-8)),
                             get(section, "cooldown", 0))
end

function parse_magic_potential_files(magic_section, system_params_list::Vector{SystemParameters})
    potential_entries = get(magic_section, "potentials", [])
    isempty(potential_entries) && return String[]

    potential_by_system = Dict{String, String}()
    for entry in potential_entries
        potential_by_system[entry["system"]] = entry["file"]
    end

    return [potential_by_system[system_params.system_name] for system_params in system_params_list]
end

function parameters_init()
    # Handle command line arguments
    if length(ARGS) == 0
        error("No arguments provided. Please provide a TOML config file path")
    elseif !endswith(ARGS[1], ".toml")
        error("Only TOML files are supported. Please provide a file with .toml extension")
    end
    inputname = ARGS[1]

    # Read and parse main configuration file
    config = TOML.parsefile(inputname)

    println("-------------------------------- Configuration --------------------------------")
    println("Config file: $(inputname)")
    println()
    TOML.print(config)
    println()

    # Parse run parameters
    VALID_MODES = ("training", "pmf-pretraining", "magic-pretraining", "simulation")
    mode = config["run"]["mode"]
    if !(mode in VALID_MODES)
        throw(ArgumentError("Unknown mode '$mode'. Valid modes: $(join(VALID_MODES, ", "))"))
    end

    # Parse top-level execution parameters
    inputs_section = config["inputs"]
    output_section = config["output"]
    checkpoint_section = config["checkpoint"]
    training_section = config["training"]

    output_dir = get(output_section, "dir", "run")
    global_params = GlobalParameters(inputs_section["system_files"],
                                     inputs_section["symmetry_function_file"],
                                     mode,
                                     output_section["detail"],
                                     checkpoint_section["model_file"],
                                     checkpoint_section["optimizer_file"],
                                     training_section["adaptive_scaling"],
                                     output_dir)

    # Parse Monte Carlo parameters
    mc_section = config["monte_carlo"]
    mc_params = MonteCarloParameters(mc_section["steps"],
                                     mc_section["equilibration_steps"],
                                     mc_section["step_adjust_frequency"],
                                     mc_section["trajectory_frequency"],
                                     mc_section["sample_frequency"])

    # Parse symmetry functions
    g2_funcs, max_cutoff, scaling = parse_symmetry_functions(global_params.symmetry_function_file)

    input_layer_size = length(g2_funcs)

    # Parse neural network and training parameters
    model_section = config["model"]
    training_optimizer_section = training_section["optimizer"]
    training_lr_section = training_section["lr_scheduler"]

    # Prepend input layer size to existing neurons array
    neurons = [input_layer_size; model_section["layer_sizes"]]

    gradient_type = get(training_section, "loss", "mse")
    if !(gradient_type in ("mse", "mae"))
        throw(ArgumentError("Unknown training.loss '$gradient_type'. Valid values: mse, mae"))
    end

    nn_params = NeuralNetParameters(g2_funcs,
                                    max_cutoff,
                                    scaling,
                                    neurons,
                                    model_section["activations"],
                                    model_section["bias"],
                                    training_section["iterations"],
                                    Float32(training_section["regularization"]),
                                    gradient_type,
                                    OptimizerConfig(training_optimizer_section["name"],
                                                    Float32(training_optimizer_section["learning_rate"]),
                                                    Float32(training_optimizer_section["momentum"]),
                                                    Float32(training_optimizer_section["decay_rates"][1]),
                                                    Float32(training_optimizer_section["decay_rates"][2])),
                                    parse_lr_scheduler(training_lr_section))

    # Parse pre-training parameters
    pt_section = config["pretraining"]
    pt_optimizer_section = pt_section["optimizer"]
    pt_lr_section = pt_section["lr_scheduler"]
    pt_gradient_type = get(pt_section, "loss", "mse")
    if !(pt_gradient_type in ("mse", "mae"))
        throw(ArgumentError("Unknown pretraining.loss '$pt_gradient_type'. Valid values: mse, mae"))
    end

    pretrain_params = PreTrainingParameters(pt_section["steps"],
                                            pt_section["batch_size"],
                                            pt_section["output_frequency"],
                                            Float32(pt_section["regularization"]),
                                            OptimizerConfig(pt_optimizer_section["name"],
                                                            Float32(pt_optimizer_section["learning_rate"]),
                                                            Float32(pt_optimizer_section["momentum"]),
                                                            Float32(pt_optimizer_section["decay_rates"][1]),
                                                            Float32(pt_optimizer_section["decay_rates"][2])),
                                            get(pt_section, "use_diff_gradient", false),
                                            get(pt_section, "move_all_particles", false),
                                            pt_gradient_type,
                                            get(pt_section, "save_frequency", 50),
                                            get(pt_section, "output_prefix", "pt"),
                                            output_dir,
                                            parse_lr_scheduler(pt_lr_section))

    # Parse magic pre-training parameters
    magic_section = get(config, "magic_pretraining", Dict{String, Any}())

    # Parse system parameters for each system file
    system_params_list = [parse_system_parameters(system_file) for system_file in global_params.system_files]
    magic_potential_files = mode == "magic-pretraining" ?
                            parse_magic_potential_files(magic_section, system_params_list) :
                            String[]
    magic_params = MagicPreTrainingParameters(magic_potential_files)

    # Create output directory
    mkpath(global_params.output_dir)

    return global_params, mc_params, nn_params, pretrain_params, magic_params, system_params_list
end
