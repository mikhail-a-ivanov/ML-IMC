using ..ML_IMC

function simulate!(model::Flux.Chain,
                   global_params::GlobalParameters,
                   mc_params::MonteCarloParameters,
                   nn_params::NeuralNetParameters,
                   system_params::SystemParameters)

    # Create input for each worker
    model_weights, _ = Flux.destructure(model)
    model_weights = Float32.(model_weights)
    input = MonteCarloSampleInput(global_params, mc_params, nn_params, system_params, model_weights)
    inputs = fill(input, nworkers())

    # Run parallel Monte Carlo sampling
    outputs = pmap(mcsample!, inputs)

    # Collect system statistics
    system_outputs,
    system_losses = collect_system_averages(outputs, nothing, [system_params], global_params,
                                            nothing, nothing, 0.0f0, 1, mc_params.steps)

    # Save simulation results
    system_name = system_params.system_name
    od = global_params.output_dir
    try
        write_rdf(joinpath(od, "RDFNN-$(system_name).dat"), system_outputs[1].descriptor, system_params)
        write_energies(joinpath(od, "energies-$(system_name).dat"), system_outputs[1].energies, mc_params,
                       system_params, 1)
    catch e
        @error "Failed to save simulation results" exception=e system=system_name
        rethrow()
    end
end
