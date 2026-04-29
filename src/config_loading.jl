using ..ML_IMC

function parse_symmetry_functions(filename::String)
    symm_data = TOML.parsefile(filename)

    println("----------------------------- Symmetry Functions ------------------------------")
    println("Symmetry Functions file: $(filename)")
    println()
    TOML.print(symm_data)
    println()

    g2_functions = Vector{G2}()
    scaling_factor = get(symm_data, "scaling", 1.0)
    max_cutoff = 0.0

    if haskey(symm_data, "G2")
        for params in symm_data["G2"]
            eta, rcutoff, rshift, norm = params
            g2 = G2(eta, rcutoff, rshift, norm)
            push!(g2_functions, g2)
            max_cutoff = max(max_cutoff, rcutoff)
        end
    end

    return g2_functions, max_cutoff, scaling_factor
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
    bin_width = infer_bin_width(bins)

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

    # Validate mode
    VALID_MODES = ("training", "pmf-pretraining", "magic-pretraining", "simulation")
    mode = config["global"]["mode"]
    if !(mode in VALID_MODES)
        throw(ArgumentError("Unknown mode '$mode'. Valid modes: $(join(VALID_MODES, ", "))"))
    end

    # Parse global parameters
    output_dir = get(config["global"], "output_dir", "run")
    global_params = GlobalParameters(config["global"]["system_files"],
                                     config["global"]["symmetry_function_file"],
                                     mode,
                                     config["global"]["output_mode"],
                                     config["global"]["model_file"],
                                     config["global"]["gradients_file"],
                                     config["global"]["optimizer_file"],
                                     config["global"]["adaptive_scaling"],
                                     output_dir)

    # Parse Monte Carlo parameters
    mc_params = MonteCarloParameters(config["monte_carlo"]["steps"],
                                     config["monte_carlo"]["equilibration_steps"],
                                     config["monte_carlo"]["step_adjust_frequency"],
                                     config["monte_carlo"]["trajectory_output_frequency"],
                                     config["monte_carlo"]["output_frequency"])

    # Parse symmetry functions
    g2_funcs, max_cutoff, scaling = parse_symmetry_functions(global_params.symmetry_function_file)

    input_layer_size = length(g2_funcs)

    # Parse neural network parameters
    nn_section = config["neural_network"]

    # Prepend input layer size to existing neurons array
    neurons = [input_layer_size; nn_section["neurons"]]

    gradient_type = get(nn_section, "gradient_type", "mse")
    if !(gradient_type in ("mse", "mae"))
        throw(ArgumentError("Unknown gradient_type '$gradient_type'. Valid values: mse, mae"))
    end

    nn_params = NeuralNetParameters(g2_funcs,
                                    max_cutoff,
                                    scaling,
                                    neurons,
                                    nn_section["activations"],
                                    nn_section["bias"],
                                    nn_section["iterations"],
                                    nn_section["regularization"],
                                    gradient_type,
                                    nn_section["optimizer"],
                                    nn_section["learning_rate"],
                                    nn_section["momentum"],
                                    nn_section["decay_rates"][1],
                                    nn_section["decay_rates"][2],
                                    LRSchedulerConfig(get(nn_section, "lr_warmup_epochs", 0),
                                                      get(nn_section, "lr_warmup_start", 1.0e-7),
                                                      get(nn_section, "lr_patience", 50),
                                                      get(nn_section, "lr_factor", 0.5),
                                                      get(nn_section, "lr_min", 1.0e-8),
                                                      get(nn_section, "lr_cooldown", 0)))

    # Parse pre-training parameters
    pt_section = config["pretraining"]
    pt_gradient_type = get(pt_section, "gradient_type", "mse")
    if !(pt_gradient_type in ("mse", "mae"))
        throw(ArgumentError("Unknown pretraining.gradient_type '$pt_gradient_type'. Valid values: mse, mae"))
    end

    pretrain_params = PreTrainingParameters(pt_section["steps"],
                                            pt_section["batch_size"],
                                            pt_section["output_frequency"],
                                            pt_section["regularization"],
                                            pt_section["optimizer"],
                                            pt_section["learning_rate"],
                                            pt_section["momentum"],
                                            pt_section["decay_rates"][1],
                                            pt_section["decay_rates"][2],
                                            get(pt_section, "use_diff_gradient", false),
                                            get(pt_section, "use_all_particles", true),
                                            pt_gradient_type,
                                            get(pt_section, "save_frequency", 50),
                                            get(pt_section, "output_prefix", "pt"),
                                            output_dir,
                                            LRSchedulerConfig(get(pt_section, "lr_warmup_epochs", 0),
                                                              get(pt_section, "lr_warmup_start", 1.0e-7),
                                                              get(pt_section, "lr_patience", 50),
                                                              get(pt_section, "lr_factor", 0.5),
                                                              get(pt_section, "lr_min", 1.0e-8),
                                                              get(pt_section, "lr_cooldown", 0)))

    # Parse magic pre-training parameters
    magic_section = get(config, "magic_pretraining", Dict{String, Any}())
    magic_params = MagicPreTrainingParameters(get(magic_section, "model_file", "none"),
                                              get(magic_section, "potential_files", String[]),
                                              get(magic_section, "use_diff_gradient", true),
                                              get(magic_section, "use_all_particles", false))

    # Parse system parameters for each system file
    system_params_list = [parse_system_parameters(system_file) for system_file in global_params.system_files]

    # Validate magic-pretraining configuration
    if global_params.mode == "magic-pretraining"
        if length(magic_params.potential_files) != length(system_params_list)
            throw(ArgumentError("Magic pretraining requires one potential file per system. " *
                                "Got $(length(magic_params.potential_files)) potential files and $(length(system_params_list)) systems."))
        end
        for pf in magic_params.potential_files
            check_file(pf)
        end
        if magic_params.model_file != "none"
            check_file(magic_params.model_file)
        end
    end

    # Create output directory
    mkpath(global_params.output_dir)

    return global_params, mc_params, nn_params, pretrain_params, magic_params, system_params_list
end
