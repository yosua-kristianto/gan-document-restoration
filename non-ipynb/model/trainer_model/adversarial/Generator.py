from tensorlayerx.nn import Module

class NetWithLoss_Adv_G(Module):
    def __init__(self, D_net, G_net, vgg, loss_fn1, loss_fn2):
        super(NetWithLoss_Adv_G, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.vgg = vgg
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def forward(self, lr, hr):
        # Generated image
        generated_image_tensor = self.G_net(lr)

        # Determinator judgement
        discriminator_logits_for_generated_image = self.D_net(generated_image_tensor)

        feature_fake = self.vgg((generated_image_tensor + 1) / 2.)
        feature_real = self.vgg((hr + 1) / 2.)

        # Adversarial Loss
        g_gan_loss = self.loss_fn1(discriminator_logits_for_generated_image, tlx.ones_like(discriminator_logits_for_generated_image))
        g_gan_loss = tlx.ops.reduce_mean(g_gan_loss)

        # Content Loss
        mse_loss = self.loss_fn2(generated_image_tensor, hr)
        vgg_loss = 2e-6 * self.loss_fn2(feature_fake, feature_real)

        g_loss = mse_loss + vgg_loss + g_gan_loss
        return g_loss