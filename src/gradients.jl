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

function initialize_cross_accumulators(system_params::SystemParameters,
                                       model::Flux.Chain)::Matrix{Float32}
    model_params, re = Flux.destructure(model)
    num_params = length(model_params)
    cross_accumulators_shape = (system_params.n_bins, num_params)
    cross_accumulators = zeros(Float32, cross_accumulators_shape)
    return cross_accumulators
end

function flatten_model_gradient!(buffer::Vector{T}, grad) where {T <: AbstractFloat}
    fill!(buffer, zero(T))
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
                                    rdf_sample::Vector{T},
                                    model::Chain,
                                    flat_grad_buffer::Vector{T})::Matrix{T} where {T <: AbstractFloat}
    energy_gradients = compute_energy_gradients(symm_func_matrix, model)
    flatten_model_gradient!(flat_grad_buffer, energy_gradients)
    BLAS.ger!(one(T), rdf_sample, flat_grad_buffer, cross_accumulators)
    return cross_accumulators
end

function compute_loss_gradients(mean_rdf_energy_grad::Matrix{T},
                                mean_energy_grad::Vector{T},
                                mean_rdf::Vector{T},
                                rdf_target::Vector{T},
                                system_params::SystemParameters,
                                nn_params::NeuralNetParameters)::Vector{T} where {T <: AbstractFloat}
    # Derivation of the loss gradient.
    #
    # The system samples the canonical ensemble with energy E_θ(x), where x is a
    # configuration and θ is the vector of neural-network parameters. The
    # Boltzmann probability of x is
    #
    #     P_θ(x) = exp(-β E_θ(x)) / Z(θ),     Z(θ) = ∫ exp(-β E_θ(x)) dx.
    #
    # Let g_i(x) be the contribution of x to the i-th RDF bin and
    # ḡ_i(θ) = <g_i> = ∫ g_i(x) P_θ(x) dx its ensemble average. Using
    # ∂(ln Z)/∂θ_p = -β <∂E/∂θ_p> one obtains
    #
    #     ∂P_θ(x)/∂θ_p = -β P_θ(x) ( ∂E/∂θ_p - <∂E/∂θ_p> ),
    #
    # and inserting this under the integral yields the fluctuation–response
    # identity
    #
    #     ∂ḡ_i/∂θ_p = -β ( <g_i ∂E/∂θ_p> - <g_i> <∂E/∂θ_p> ).
    #
    # The loss L depends on θ only through ḡ, hence the chain rule gives
    #
    #     ∂L/∂θ_p = Σ_i (∂L/∂ḡ_i) (∂ḡ_i/∂θ_p)
    #             = -β Σ_i (∂L/∂ḡ_i) ( <g_i ∂E/∂θ_p> - <g_i> <∂E/∂θ_p> ).
    #
    # Mapping the averages to the variables used below,
    #
    #     mean_rdf_energy_grad[i, p]  = <g_i ∂E/∂θ_p>
    #     mean_energy_grad[p]         = <∂E/∂θ_p>
    #     mean_rdf[i]                 = <g_i>
    #     loss_grad_rdf[i]            = ∂L/∂ḡ_i,
    #
    # the gradient takes the compact form
    #
    #     ∂L/∂θ = -β ( mean_rdf_energy_grad' * loss_grad_rdf
    #                  - (mean_rdf · loss_grad_rdf) * mean_energy_grad ),
    #
    # which avoids assembling the full n_bins × n_params Jacobian ∂ḡ/∂θ.
    rdf_residual = mean_rdf - rdf_target
    inv_n_bins = one(T) / T(length(rdf_residual))
    loss_grad_rdf = if nn_params.gradient_type == "mae"
        sign.(rdf_residual) .* inv_n_bins
    else
        (T(2) * inv_n_bins) .* rdf_residual
    end

    joint_term = mean_rdf_energy_grad' * loss_grad_rdf
    product_term = dot(mean_rdf, loss_grad_rdf) .* mean_energy_grad
    loss_grad_params = -T(system_params.beta) .* (joint_term .- product_term)

    return loss_grad_params
end
