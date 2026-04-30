using ..ML_IMC

function init_optimizer(params::Union{NeuralNetParameters, PreTrainingParameters})
    OPTIMIZER_MAP = Dict("Momentum" => Momentum,
                         "Descent" => Descent,
                         "Nesterov" => Nesterov,
                         "RMSProp" => RMSProp,
                         "Adam" => Adam,
                         "RAdam" => RAdam,
                         "AdaMax" => AdaMax,
                         "AdaGrad" => AdaGrad,
                         "AdaDelta" => AdaDelta,
                         "AMSGrad" => AMSGrad,
                         "NAdam" => NAdam,
                         "AdamW" => AdamW,
                         "OAdam" => OAdam,
                         "AdaBelief" => AdaBelief)

    opt_config = params.optimizer_config

    optimizer_func = get(OPTIMIZER_MAP, opt_config.name, nothing)
    if isnothing(optimizer_func)
        throw(ArgumentError("Unknown optimizer: $(opt_config.name). Supported: $(join(keys(OPTIMIZER_MAP), ", "))"))
    end

    if optimizer_func in (Momentum, Nesterov, RMSProp)
        return optimizer_func(opt_config.learning_rate, opt_config.momentum)
    elseif optimizer_func in (Adam, RAdam, AdaMax, AMSGrad, NAdam, AdamW, OAdam, AdaBelief)
        return optimizer_func(opt_config.learning_rate, (opt_config.decay_1, opt_config.decay_2))
    else
        return optimizer_func(opt_config.learning_rate)
    end
end

function build_network(nn_params::NeuralNetParameters)
    return [(nn_params.neurons[i - 1], nn_params.neurons[i],
             getfield(Flux, Symbol(nn_params.activations[i - 1])))
            for i in 2:length(nn_params.neurons)]
end

function build_chain(nn_params::NeuralNetParameters, layers...)
    return Chain([nn_params.bias ? Dense(layer...) : Dense(layer..., bias=false)
                  for layer in layers]...)
end

function model_init(nn_params::NeuralNetParameters)
    network = build_network(nn_params)
    model = build_chain(nn_params, network...)
    model = f64(model)

    return model
end

function update_model!(model::Chain, opt_state, loss_gradients)
    Flux.update!(opt_state, model, loss_gradients)

    return model
end
