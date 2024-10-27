using TOML
using Chemfiles

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
        println("Trying to read input data from configs/config.toml")
        "configs/config.toml"
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
