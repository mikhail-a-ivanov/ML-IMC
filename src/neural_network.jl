using Flux

function build_network(nn_params::NeuralNetParameters)
    return [(nn_params.neurons[i - 1], nn_params.neurons[i],
             getfield(Flux, Symbol(nn_params.activations[i - 1])))
            for i in 2:length(nn_params.neurons)]
end

function build_chain(nn_params::NeuralNetParameters, layers...)
    return Chain([nn_params.bias ? Dense(layer...) : Dense(layer..., bias=false)
                  for layer in layers]...)
end

function init_optimizer(params::Union{NeuralNetParameters, PreTrainingParameters})
    function get_rate(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.learning_rate : params.learning_rate
    end

    function get_momentum(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.momentum : params.momentum
    end

    function get_decay1(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.decay_1 : params.decay_1
    end

    function get_decay2(params::Union{NeuralNetParameters, PreTrainingParameters})
        return params isa NeuralNetParameters ? params.decay_2 : params.decay_2
    end

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

    optimizer_name = params isa NeuralNetParameters ? params.optimizer : params.optimizer
    optimizer_func = get(OPTIMIZER_MAP, optimizer_name, Descent)

    if optimizer_func in (Momentum, Nesterov, RMSProp)
        return optimizer_func(get_rate(params), get_momentum(params))
    elseif optimizer_func in (Adam, RAdam, AdaMax, AMSGrad, NAdam, AdamW, OAdam, AdaBelief)
        return optimizer_func(get_rate(params), (get_decay1(params), get_decay2(params)))
    else
        return optimizer_func(get_rate(params))
    end
end

function model_init(nn_params::NeuralNetParameters)
    network = build_network(nn_params)
    model = build_chain(nn_params, network...)
    model = f64(model)

    return model
end

function update_model!(model::Chain,
                       optimizer::Flux.Optimise.AbstractOptimiser,
                       loss_gradients::Vector{<:AbstractArray{T}}) where {T <: AbstractFloat}
    for (gradient, parameter) in zip(loss_gradients, Flux.params(model))
        Flux.Optimise.update!(optimizer, parameter, gradient)
    end

    return model
end
