using ..ML_IMC

function parse_symmetry_functions(filename::String)
    symm_data = TOML.parsefile(filename)

    println("----------------------------- Symmetry Functions ------------------------------")
    println("Symmetry Functions file: $(filename)")
    println()
    TOML.print(symm_data)
    println()

    g2_functions = Vector{G2}()
    g3_functions = Vector{G3}()
    g9_functions = Vector{G9}()
    scaling_factor = get(symm_data, "scaling", 1.0)
    max_cutoff = 0.0

    # Parse G2 functions
    if haskey(symm_data, "G2")
        for params in symm_data["G2"]
            eta, rcutoff, rshift, norm = params
            g2 = G2(eta, rcutoff, rshift, norm)
            push!(g2_functions, g2)
            max_cutoff = max(max_cutoff, rcutoff)
        end
    end

    # Parse G3 functions (if uncommented in config)
    if haskey(symm_data, "G3")
        for params in symm_data["G3"]
            eta, lambda, zeta, rcutoff, rshift = params
            g3 = G3(eta, lambda, zeta, rcutoff, rshift)
            push!(g3_functions, g3)
            max_cutoff = max(max_cutoff, rcutoff)
        end
    end

    # Parse G9 functions (if uncommented in config)
    if haskey(symm_data, "G9")
        for params in symm_data["G9"]
            eta, lambda, zeta, rcutoff, rshift = params
            g9 = G9(eta, lambda, zeta, rcutoff, rshift)
            push!(g9_functions, g9)
            max_cutoff = max(max_cutoff, rcutoff)
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

    return global_params, mc_params, nn_params, pretrain_params, system_params_list
end
