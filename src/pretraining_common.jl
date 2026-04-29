using ..ML_IMC

function compute_pretraining_gradient!(e_nn::T, e_ref::T, Δe_nn::T, Δe_ref::T,
                                       symm1::AbstractMatrix{T}, symm2::AbstractMatrix{T},
                                       model::Chain, pretrain_params::PreTrainingParameters,
                                       use_diff_gradient::Bool;
                                       norm_factor::T=T(1))::Any where {T <: AbstractFloat}
    model_params = Flux.trainables(model)
    mp, _ = Flux.destructure(model_params)
    reg_gradient = @. mp * 2 * pretrain_params.regularization

    if !use_diff_gradient
        grad_final = compute_energy_gradients(symm2, model)
        diff = (e_nn - e_ref) / norm_factor
        gradient_scale = pretrain_params.gradient_type == "mae" ? sign(diff) : T(2) * diff
        flat_grad, restructure = Flux.destructure(grad_final)
        loss_gradient = @. gradient_scale * flat_grad / norm_factor
        return restructure(loss_gradient + reg_gradient)
    else
        grad1 = compute_energy_gradients(symm1, model)
        grad2 = compute_energy_gradients(symm2, model)
        diff = (Δe_nn - Δe_ref) / norm_factor
        gradient_scale = pretrain_params.gradient_type == "mae" ? sign(diff) : T(2) * diff
        flat_grad1, restructure = Flux.destructure(grad1)
        flat_grad2, _ = Flux.destructure(grad2)
        loss_gradient = @. gradient_scale * (flat_grad2 - flat_grad1) / norm_factor
        return restructure(loss_gradient + reg_gradient)
    end
end

function log_pretraining_summary(io::IO, epoch::Int, mean_diff_mae, mean_abs_mae,
                                 grad_norm, lr)
    println(io, @sprintf("%d,%.17e,%.17e,%.17e,%.17e", epoch, mean_diff_mae, mean_abs_mae,
                         grad_norm, lr))
end

function mean_gradient(batch_gradients::Vector{Any})
    n = length(batch_gradients)
    first_flat_grad, grad_restructure = Flux.destructure(batch_gradients[1])
    for grad in batch_gradients[2:end]
        flat_grad, _ = Flux.destructure(grad)
        first_flat_grad .+= flat_grad
    end
    first_flat_grad ./= n
    return first_flat_grad, grad_restructure
end
