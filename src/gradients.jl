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
                                  model::Flux.Chain) where {T <: AbstractFloat}
    gs = gradient(compute_system_total_energy_scalar, symm_func_matrix, model)
    energy_gradients = gs[2]

    return energy_gradients
end

function compute_cross_correlation(descriptor::Vector{T}, energy_gradients)::Matrix{T} where {T <: AbstractFloat}
    flat_energy_grad, gradient_restructure = Flux.destructure(energy_gradients)
    cross_correlations = descriptor * transpose(flat_energy_grad)

    return cross_correlations
end

function initialize_cross_accumulators(system_params::SystemParameters,
                                       model::Flux.Chain)::Matrix{Float64}
    model_params, re = Flux.destructure(model)
    num_params = length(model_params)
    cross_accumulators_shape = (system_params.n_bins, num_params)
    cross_accumulators = zeros(cross_accumulators_shape)
    return cross_accumulators
end

function update_cross_accumulators!(cross_accumulators::Matrix{T},
                                    symm_func_matrix::Matrix{T},
                                    descriptor::Vector{T},
                                    model::Chain)::Matrix{T} where {T <: AbstractFloat}
    energy_gradients = compute_energy_gradients(symm_func_matrix, model)
    new_cross_correlations = compute_cross_correlation(descriptor, energy_gradients)
    cross_accumulators .+= new_cross_correlations
    return cross_accumulators
end

function compute_ensemble_correlation(symm_func_matrix::Matrix{T},
                                      descriptor::Vector{T},
                                      model::Chain)::Matrix{T} where {T <: AbstractFloat}
    energy_gradients = compute_energy_gradients(symm_func_matrix, model)
    ensemble_correlations = compute_cross_correlation(descriptor, energy_gradients)
    return ensemble_correlations
end

function compute_descriptor_gradients(cross_accumulators::Matrix{T},
                                      ensemble_correlations::Matrix{T},
                                      system_params::SystemParameters)::Matrix{T} where {T <: AbstractFloat}
    descriptor_gradients = -system_params.beta .* (cross_accumulators - ensemble_correlations)
    return descriptor_gradients
end

function compute_loss_gradients(cross_accumulators::Matrix{T},
                                symm_func_matrix::Matrix{T},
                                descriptor_nn::Vector{T},
                                descriptor_ref::Vector{T},
                                model::Flux.Chain,
                                system_params::SystemParameters,
                                nn_params::NeuralNetParameters)::Vector{T} where {T <: AbstractFloat}
    flat_params, _ = Flux.destructure(model)

    # Calculate correlations and their gradients for descriptor computation
    ensemble_correlations = compute_ensemble_correlation(symm_func_matrix, descriptor_nn, model)
    descriptor_gradients = compute_descriptor_gradients(cross_accumulators, ensemble_correlations, system_params)

    # MSE loss gradient computation
    diff = descriptor_nn - descriptor_ref
    dLdS = @. (2 / length(diff)) * diff
    # dLdS = @. (1 / length(diff)) * sign(diff)  # Градиент для MAE

    # Combine descriptor gradients with regularization
    param_gradients = descriptor_gradients' * dLdS  # (num_params × n_bins) * (n_bins × 1) = (num_params × 1)
    param_gradients .+= 2 * nn_params.regularization .* flat_params

    return param_gradients
end
