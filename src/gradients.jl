using ..ML_IMC

function compute_system_total_energy_scalar(symm_func_matrix::AbstractMatrix{T},
                                            model::Flux.Chain) where {T <: AbstractFloat}
    return sum(model(symm_func_matrix))
end

function init_system_energies_vector(symm_func_matrix::AbstractMatrix{T}, model::Flux.Chain) where {T <: AbstractFloat}
    return vec(model(symm_func_matrix))
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
                                       model::Flux.Chain)::Matrix{Float32}
    model_params, re = Flux.destructure(model)
    num_params = length(model_params)
    cross_accumulators_shape = (system_params.n_bins, num_params)
    cross_accumulators = zeros(Float32, cross_accumulators_shape)
    return cross_accumulators
end

function flatten_model_gradient!(buffer::Vector{T}, grad) where {T <: AbstractFloat}
    offset = 1
    if hasproperty(grad, :layers)
        for layer_grad in grad.layers
            if isnothing(layer_grad)
                continue
            end
            if hasproperty(layer_grad, :weight) && !isnothing(layer_grad.weight)
                w = layer_grad.weight
                n = length(w)
                buffer[offset:(offset + n - 1)] .= vec(w)
                offset += n
            end
            if hasproperty(layer_grad, :bias) && !isnothing(layer_grad.bias)
                b = layer_grad.bias
                n = length(b)
                buffer[offset:(offset + n - 1)] .= vec(b)
                offset += n
            end
        end
    end
    return buffer
end

function update_cross_accumulators!(cross_accumulators::Matrix{T},
                                    symm_func_matrix::Matrix{T},
                                    descriptor::Vector{T},
                                    model::Chain,
                                    flat_grad_buffer::Vector{T})::Matrix{T} where {T <: AbstractFloat}
    energy_gradients = compute_energy_gradients(symm_func_matrix, model)
    flatten_model_gradient!(flat_grad_buffer, energy_gradients)
    BLAS.ger!(one(T), descriptor, flat_grad_buffer, cross_accumulators)
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

    # MSE/MAE loss gradient computation
    diff = descriptor_nn - descriptor_ref
    dLdS = if nn_params.gradient_type == "mae"
        @. sign(diff) / T(length(diff))
    else
        @. (T(2) / T(length(diff))) * diff
    end

    # Combine descriptor gradients with regularization
    param_gradients = descriptor_gradients' * dLdS  # (num_params × n_bins) * (n_bins × 1) = (num_params × 1)
    param_gradients .+= T(2) * T(nn_params.regularization) .* flat_params

    return param_gradients
end
