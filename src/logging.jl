using ..ML_IMC

# Constants for RST-style formatting
const SECTION_CHARS = ('=', '-', '~', '"', '^')
const INDENT = "    "

"""
Creates a section header with underline
"""
function format_section(text::String, level::Int=1)
    char = SECTION_CHARS[level]
    underline = char^length(text)
    return "\n$text\n$underline\n"
end

"""
Creates an indented block of code or data
"""
function format_code_block(text::String)
    # Split text into lines and indent each line
    indented = join(INDENT * line for line in split(text, '\n'))
    return "::\n\n$indented\n"
end

"""
Formats a list item with proper indentation
"""
function format_item(text::String, indent_level::Int=0)
    prefix = INDENT^indent_level
    return "$prefix* $text"
end

# Main logging functions
function log_section(text::String)
    println(format_section(text, 1))
end

function log_subsection(text::String)
    println(format_section(text, 2))
end

function log_subsubsection(text::String)
    println(format_section(text, 3))
end

function log_item(text::String, indent_level::Int=0)
    println(format_item(text, indent_level))
end

function log_code_block(text::String)
    println(format_code_block(text))
end

function log_global_parameters(params::GlobalParameters)
    log_subsubsection("Global Parameters")

    # System Files
    log_item("System Configuration")
    println("* Input Files::")
    for (i, file) in enumerate(params.system_files)
        println(INDENT * "System $i: $(basename(file))")
    end
    println(INDENT * "Symmetry functions: $(basename(params.symmetry_function_file))")
    println()

    # Operation Mode
    log_item("Operation Mode")
    println(INDENT * "Mode: $(params.mode)")
    println(INDENT * "Output verbosity: $(params.output_mode)")
    println()

    # Model Configuration
    log_item("Model Configuration")
    println("* Files::")
    println(INDENT * "Model: $(params.model_file == "none" ? "new model" : basename(params.model_file))")
    println(INDENT * "Gradients: $(params.gradients_file == "none" ? "not provided" : basename(params.gradients_file))")
    println(INDENT * "Optimizer: $(params.optimizer_file == "none" ? "not provided" : basename(params.optimizer_file))")
    println()

    # Training Settings
    log_item("Training Settings")
    println(INDENT * "Adaptive gradient scaling: $(params.adaptive_scaling)")
    println()
end

function log_model_info(model::Chain, nn_params::NeuralNetParameters)
    log_subsubsection("Model Architecture")

    # Format model layers
    layers_str = join(string.(model.layers), "\n")
    log_code_block(layers_str)

    log_item("Layers: $(length(nn_params.neurons))")
    log_item("Neurons: $(nn_params.neurons)")
    log_item("Parameters: $(sum(length.(Flux.params(model))))")
    log_item("Bias: $(nn_params.bias ? "enabled" : "disabled")")
    println()
end

function log_symmetry_functions_info(nn_params::NeuralNetParameters)
    log_subsubsection("Symmetry Functions")

    # G2 Functions
    if !isempty(nn_params.g2_functions)
        log_item("G2 Functions")
        println("\n* Parameters (η [Å⁻²], rcutoff [Å], rshift [Å])::")
        for g2 in nn_params.g2_functions
            println(INDENT * "($(g2.eta), $(g2.rcutoff), $(g2.rshift))")
        end
        println()
    end

    # G3 Functions
    if !isempty(nn_params.g3_functions)
        log_item("G3 Functions")
        println("\n* Parameters (η [Å⁻²], λ, ζ, rcutoff [Å], rshift [Å])::")
        for g3 in nn_params.g3_functions
            println(INDENT * "($(g3.eta), $(g3.lambda), $(g3.zeta), " *
                    "$(g3.rcutoff), $(g3.rshift))")
        end
        println()
    end

    # G9 Functions
    if !isempty(nn_params.g9_functions)
        log_item("G9 Functions")
        println("\n* Parameters (η [Å⁻²], λ, ζ, rcutoff [Å], rshift [Å])::")
        for g9 in nn_params.g9_functions
            println(INDENT * "($(g9.eta), $(g9.lambda), $(g9.zeta), " *
                    "$(g9.rcutoff), $(g9.rshift))")
        end
        println()
    end

    # Global parameters
    log_item("Global Parameters")
    println(INDENT * "Maximum cutoff distance: $(nn_params.max_distance_cutoff) Å")
    println(INDENT * "Symmetry function scaling: $(nn_params.symm_function_scaling)")
    println()
end

function log_training_config(global_params::GlobalParameters, mc_params::MonteCarloParameters)
    log_subsubsection("Training Configuration")
    log_item("Mode: $(global_params.mode)")
    log_item("Reference systems: $(length(global_params.system_files))")
    log_item("Workers: $(nworkers())")
    log_item("Steps: $(mc_params.steps/1e6)M")
    log_item("Equilibration: $(mc_params.equilibration_steps/1e6)M")
    println()
end

function log_optimizer_info(optimizer::Flux.Optimise.AbstractOptimiser)
    log_subsubsection("Optimizer Configuration")

    # Get optimizer type name without module prefix
    optimizer_type = split(string(typeof(optimizer)), ".")[end]
    log_item("Type: $optimizer_type")
    println()

    # Parameter descriptions with improved naming
    param_descriptions = Dict(:eta => "Learning rate",
                              :beta => "Decay rates",
                              :beta1 => "First moment decay rate",
                              :beta2 => "Second moment decay rate",
                              :rho => "Momentum coefficient",
                              :gamma => "Learning rate decay",
                              :alpha => "RMSprop smoothing coefficient",
                              :epsilon => "Numerical stability constant",
                              :clip => "Gradient clipping threshold",
                              :momentum => "Momentum coefficient")

    # Group parameters by category
    println("* Learning Parameters::")
    for name in fieldnames(typeof(optimizer))
        # Skip internal state fields
        name in (:state, :velocity, :params) && continue

        value = getfield(optimizer, name)
        description = get(param_descriptions, name, string(name))

        # Format different types of values
        formatted_value = if isa(value, Tuple)
            join([string(round(v, digits=6)) for v in value], ", ")
        elseif isa(value, AbstractFloat)
            string(round(value, digits=6))
        else
            string(value)
        end

        # Print with proper indentation and alignment
        println(INDENT * "$(description): $(formatted_value)")
    end
    println()
end
