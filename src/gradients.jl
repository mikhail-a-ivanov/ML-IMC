using ..ML_IMC

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

    # NOTE: The order of difference is very important
    dLdS = @. 2 * (descriptor_nn - descriptor_ref)

    loss_gradients = Vector{AbstractArray{T}}(undef, length(Flux.params(model)))

    for (i, (gradient, parameters)) in enumerate(zip(descriptor_gradients, Flux.params(model)))
        loss_gradient = reshape(dLdS' * gradient, size(parameters))
        reg_loss_gradient = @. 2 * nn_params.regularization * parameters
        loss_gradients[i] = loss_gradient .+ reg_loss_gradient
    end

    return loss_gradients
end
