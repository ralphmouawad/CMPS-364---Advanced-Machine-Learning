import numpy as np

def sgd(parameters, parameter_gradients, optimizer_config=None):
    if optimizer_config is None:
        optimizer_config = {}
    optimizer_config.setdefault("learning_rate", 1e-2)

    step_size = optimizer_config["learning_rate"]
    parameters -= step_size * parameter_gradients
    return parameters, optimizer_config

def sgd_momentum(parameters, parameter_gradients, optimizer_config=None):
    if optimizer_config is None:
        optimizer_config = {}
    optimizer_config.setdefault("learning_rate", 1e-2)
    optimizer_config.setdefault("momentum", 0.9)
    accumulated_velocity = optimizer_config.get("velocity", np.zeros_like(parameters))

    step_size = optimizer_config["learning_rate"]
    momentum_coefficient = optimizer_config["momentum"]
    accumulated_velocity = momentum_coefficient * accumulated_velocity - step_size * parameter_gradients
    parameters += accumulated_velocity
    optimizer_config["velocity"] = accumulated_velocity

    return parameters, optimizer_config

def rmsprop(parameters, parameter_gradients, optimizer_config=None):
    if optimizer_config is None:
        optimizer_config = {}
    optimizer_config.setdefault("learning_rate", 1e-2)
    optimizer_config.setdefault("decay_rate", 0.99)
    optimizer_config.setdefault("epsilon", 1e-8)
    optimizer_config.setdefault("squared_grad_cache", np.zeros_like(parameters))

    step_size = optimizer_config["learning_rate"]
    history_decay = optimizer_config["decay_rate"]
    smoothing_term = optimizer_config["epsilon"]
    squared_grad_cache = optimizer_config["squared_grad_cache"]

    squared_grad_cache = history_decay * squared_grad_cache + (1.0 - history_decay) * (parameter_gradients ** 2)
    parameter_update = -(step_size * parameter_gradients) / (np.sqrt(squared_grad_cache) + smoothing_term)
    parameters += parameter_update
    optimizer_config["squared_grad_cache"] = squared_grad_cache

    return parameters, optimizer_config

def adam(parameters, parameter_gradients, optimizer_config=None):
    if optimizer_config is None:
        optimizer_config = {}
    optimizer_config.setdefault("learning_rate", 1e-3)
    optimizer_config.setdefault("beta1", 0.9)
    optimizer_config.setdefault("beta2", 0.999)
    optimizer_config.setdefault("epsilon", 1e-8)
    optimizer_config.setdefault("momentum_cache", np.zeros_like(parameters))
    optimizer_config.setdefault("velocity_cache", np.zeros_like(parameters))
    optimizer_config.setdefault("iteration", 0)

    step_size = optimizer_config["learning_rate"]
    momentum_decay = optimizer_config["beta1"]
    velocity_decay = optimizer_config["beta2"]
    smoothing_term = optimizer_config["epsilon"]
    momentum_cache = optimizer_config["momentum_cache"]
    velocity_cache = optimizer_config["velocity_cache"]
    current_iteration = optimizer_config["iteration"]

    momentum_cache = momentum_decay * momentum_cache + (1 - momentum_decay) * parameter_gradients
    velocity_cache = velocity_decay * velocity_cache + (1 - velocity_decay) * (parameter_gradients ** 2)
    current_iteration += 1

    bias_corrected_step_size = (
        step_size * np.sqrt(1 - velocity_decay ** current_iteration) / 
        (1 - momentum_decay ** current_iteration)
    )
    parameter_update = momentum_cache / (np.sqrt(velocity_cache) + smoothing_term)
    parameters -= bias_corrected_step_size * parameter_update

    optimizer_config["momentum_cache"] = momentum_cache
    optimizer_config["velocity_cache"] = velocity_cache
    optimizer_config["iteration"] = current_iteration

    return parameters, optimizer_config