using Flux

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
