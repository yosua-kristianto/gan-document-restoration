hyperparameters = {
    "batch_size": 16,
    "generator_dimension": 56,
    # Initialization Training
    "init": {
        "epoch_total": 300,
        "learning_rate": 5e-2,

        "decay_step_size": 200,
        "decay_learning_rate": 1e-4,
        "decay_gamma": 1e-1,
    },

    # Adversarial Training
    "discriminator_dimension": 224, # 56 x 4
    "adv": {
        "epoch_total": 500,
        "learning_rate": 5e-2,

        "decay_step_size": 400,
        "decay_learning_rate": 1e-3,
        "decay_gamma": 1e-1
    }
}