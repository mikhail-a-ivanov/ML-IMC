using ..ML_IMC

function make_default_scheduler_config()
    return LRSchedulerConfig(0, 1.0f-7, 50, 0.5f0, 1.0f-8, 0)
end

function lr_for_epoch(config::LRSchedulerConfig, initial_lr::Float32, epoch::Int)
    if epoch <= config.warmup_epochs
        if config.warmup_epochs > 1
            return config.warmup_start_lr +
                   (initial_lr - config.warmup_start_lr) * Float32(epoch - 1) /
                   Float32(config.warmup_epochs - 1)
        else
            return initial_lr
        end
    end
    return initial_lr
end

function step_plateau!(config::LRSchedulerConfig, state::LRSchedulerState, opt_state, loss::Float32)
    if config.patience <= 0
        return state.current_lr
    end

    if state.cooldown_left > 0
        state.cooldown_left -= 1
        return state.current_lr
    end

    if loss < state.best_loss
        state.best_loss = loss
        state.bad_epochs = 0
    else
        state.bad_epochs += 1
    end

    if state.bad_epochs >= config.patience
        new_lr = state.current_lr * config.factor
        if new_lr >= config.min_lr
            state.current_lr = new_lr
            Flux.adjust!(opt_state, new_lr)
            state.bad_epochs = 0
            state.cooldown_left = config.cooldown
        end
    end

    return state.current_lr
end
