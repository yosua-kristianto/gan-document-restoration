from tensorlayerx.nn import Module

class NetWithLoss_init(Module):
    def __init__(self, generator_model, loss_fn):
        super(NetWithLoss_init, self).__init__()
        self.net = generator_model
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        out = self.net(lr)
        loss = self.loss_fn(out, hr)
        return loss

def execute_init_training():
    from tensorlayerx.optimizers.lr import StepDecay
    from tensorlayerx.optimizers import Adam
    from config import hyperparameters
    from model import Generator
    from tensorlayerx.nn import Input

    decay = StepDecay(
        learning_rate = hyperparameters["init"]["decay_learning_rate"],
        step_size = hyperparameters["init"]["decay_step_size"],
        gamma = hyperparameters["init"]["decay_gamma"],
        last_epoch = -1,
        verbose = True
    )

    optimizer = Adam(decay, hyperparameters["init"]["learning_rate"])

    generator_model = Generator()
    generator_model.init_build(
        Input(
            shape = (
                hyperparameters["batch_size"],
                3,
                hyperparameters["generator_dimension"],
                hyperparameters["generator_dimension"]
            )
        )
    )

    g_weights = generator_model.trainable_weights

    from tensorlayerx.losses import mean_squared_error
    from tensorlayerx.model import TrainOneStep

    Net_with_loss = NetWithLoss_init(generator_model = generator_model, loss_fn = mean_squared_error)
    trainer = TrainOneStep(
        Net_with_loss,
        optimizer = optimizer,
        train_weights = g_weights
    )

