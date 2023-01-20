using Flux

function reportOpt(opt)
    println("Optimizer type: $(typeof(opt))")
    println("   Parameters:")
    for name in fieldnames(typeof(opt))
        if String(name) != "state" && String(name) != "velocity"
            if String(name) == "eta"
                println("       Learning rate: $(getfield(opt, (name)))")
            elseif String(name) == "beta"
                println("       Decays: $(getfield(opt, (name)))")
            elseif String(name) == "velocity"
                println("       Velocity: $(getfield(opt, (name)))")
            elseif String(name) == "beta"
                println("       Beta: $(getfield(opt, (name)))")
            elseif String(name) == "rho"
                println("       Momentum coefficient: $(getfield(opt, (name)))")
            end
        end
    end
end

"""
function optInit(NNParms::NNparameters)

Initializes the optimizer for the training
"""
function optInit(NNParms::NNparameters)
    if NNParms.optimizer == "Momentum"
        opt = Momentum(NNParms.rate, NNParms.momentum)

    elseif NNParms.optimizer == "Descent"
        opt = Descent(NNParms.rate)

    elseif NNParms.optimizer == "Nesterov"
        opt = Nesterov(NNParms.rate, NNParms.momentum)

    elseif NNParms.optimizer == "RMSProp"
        opt = RMSProp(NNParms.rate, NNParms.momentum)

    elseif NNParms.optimizer == "Adam"
        opt = Adam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "RAdam"
        opt = RAdam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdaMax"
        opt = AdaMax(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdaGrad"
        opt = AdaGrad(NNParms.rate)

    elseif NNParms.optimizer == "AdaDelta"
        opt = AdaDelta(NNParms.rate)

    elseif NNParms.optimizer == "AMSGrad"
        opt = AMSGrad(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "NAdam"
        opt = NAdam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdamW"
        opt = AdamW(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "OAdam"
        opt = OAdam(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    elseif NNParms.optimizer == "AdaBelief"
        opt = AdaBelief(NNParms.rate, (NNParms.decay1, NNParms.decay2))

    else
        opt = Descent(NNParms.rate)
        println(
            "Unsupported type of optimizer! \n
            Default optimizer is 'Descent' \n
            For more optimizers look at: https://fluxml.ai/Flux.jl/stable/training/optimisers/ \n",
        )
    end
    return (opt)
end

"""
function optInit(preTrainParms::preTrainParameters)

Initializes the optimizer for the pre-training
"""
function optInit(preTrainParms::PreTrainParameters)
    if preTrainParms.PToptimizer == "Momentum"
        opt = Momentum(preTrainParms.PTrate, preTrainParms.PTmomentum)

    elseif preTrainParms.PToptimizer == "Descent"
        opt = Descent(preTrainParms.PTrate)

    elseif preTrainParms.PToptimizer == "Nesterov"
        opt = Nesterov(preTrainParms.PTrate, preTrainParms.PTmomentum)

    elseif preTrainParms.PToptimizer == "RMSProp"
        opt = RMSProp(preTrainParms.PTrate, preTrainParms.PTmomentum)

    elseif preTrainParms.PToptimizer == "Adam"
        opt = Adam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "RAdam"
        opt = RAdam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdaMax"
        opt = AdaMax(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdaGrad"
        opt = AdaGrad(preTrainParms.PTrate)

    elseif preTrainParms.PToptimizer == "AdaDelta"
        opt = AdaDelta(preTrainParms.PTrate)

    elseif preTrainParms.PToptimizer == "AMSGrad"
        opt = AMSGrad(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "NAdam"
        opt = NAdam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdamW"
        opt = AdamW(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "OAdam"
        opt = OAdam(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    elseif preTrainParms.PToptimizer == "AdaBelief"
        opt = AdaBelief(preTrainParms.PTrate, (preTrainParms.PTdecay1, preTrainParms.PTdecay2))

    else
        opt = Descent(preTrainParms.PTrate)
        println(
            "Unsupported type of optimizer! \n
            Default optimizer is 'Descent' \n
            For more optimizers look at: https://fluxml.ai/Flux.jl/stable/training/optimisers/ \n",
        )
    end
    return (opt)
end