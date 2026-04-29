using ..ML_IMC

function compute_pretraining_gradient!(e_nn::T, e_ref::T, Δe_nn::T, Δe_ref::T,
                                       symm1::AbstractMatrix{T}, symm2::AbstractMatrix{T},
                                       model::Chain, pretrain_params::PreTrainingParameters,
                                       use_diff_gradient::Bool)::Any where {T <: AbstractFloat}
    model_params = Flux.trainables(model)
    mp, _ = Flux.destructure(model_params)
    reg_gradient = @. mp * 2 * pretrain_params.regularization

    if !use_diff_gradient
        grad_final = compute_energy_gradients(symm2, model)
        diff = e_nn - e_ref
        gradient_scale = pretrain_params.gradient_type == "mae" ? sign(diff) : T(2) * diff
        flat_grad, restructure = Flux.destructure(grad_final)
        loss_gradient = @. gradient_scale * flat_grad
        return restructure(loss_gradient + reg_gradient)
    else
        grad1 = compute_energy_gradients(symm1, model)
        grad2 = compute_energy_gradients(symm2, model)
        diff = Δe_nn - Δe_ref
        gradient_scale = pretrain_params.gradient_type == "mae" ? sign(diff) : T(2) * diff
        flat_grad1, restructure = Flux.destructure(grad1)
        flat_grad2, _ = Flux.destructure(grad2)
        loss_gradient = @. gradient_scale * (flat_grad2 - flat_grad1)
        return restructure(loss_gradient + reg_gradient)
    end
end

function log_batch_metrics(file::Union{IO, String}, epoch, batch_iter, sys_id,
                           diff_mae, diff_mse, abs_mae, abs_mse,
                           e_nn2, e_ref2, Δe_nn, Δe_ref)
    io = file isa IO ? file : open(file, "a")
    try
        println(io,
                @sprintf("%d %d %d  %.8f %.8f  %.8f %.8f  %.8f %.8f  %.8f %.8f",
                         epoch, batch_iter, sys_id,
                         diff_mae, diff_mse,
                         abs_mae, abs_mse,
                         e_nn2, e_ref2,
                         Δe_nn, Δe_ref))
    catch e
        @warn "Failed to write to log file" exception=e
    finally
        if !(file isa IO)
            close(io)
        end
    end
end

function log_average_metrics(file::Union{IO, String}, epoch,
                             mean_mae_diff, mean_mse_diff,
                             mean_mae_abs, mean_mse_abs, mean_reg)
    io = file isa IO ? file : open(file, "a")
    try
        println(io,
                @sprintf("%d %.8f %.8f %.8f %.8f %.2e",
                         epoch,
                         mean_mae_diff, mean_mse_diff,
                         mean_mae_abs, mean_mse_abs,
                         mean_reg))
    catch e
        @warn "Failed to write average metrics to file" exception=e
    finally
        if !(file isa IO)
            close(io)
        end
    end
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
