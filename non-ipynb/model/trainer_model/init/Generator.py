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