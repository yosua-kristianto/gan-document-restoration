import tensorlayerx as tlx
import os

os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tqdm import tqdm
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
from tensorlayerx.losses import mean_squared_error

from config.index import used_weight_d, used_weight_g, training_iteration
from model.Generator import Generator
from model.Discriminator import Discriminator

tlx.set_device('GPU')

# Hyperparameters
batch_size = 16
epoch_total = 500
current_epoch = training_iteration * epoch_total
expected_last_epoch = current_epoch + epoch_total

decay = tlx.optimizers.lr.StepDecay(
    learning_rate = 1e-3,
    step_size = 2400,
    gamma = 1e-1,
    last_epoch = -1,
    verbose = True
)

generator_model = Generator()
discriminator_model = Discriminator()

generator_model.init_build(tlx.nn.Input(shape=(batch_size, 3, 56, 56)))
discriminator_model.init_build(tlx.nn.Input(shape=(16, 3, 224, 224)))

if(used_weight_g):
    generator_model.load_weights(used_weight_g, format = "npz_dict", skip = False)

if(used_weight_d):
    discriminator_model.load_weights(used_weight_d, format = "npz_dict", skip = False)

g_weights = generator_model.trainable_weights
d_weights = discriminator_model.trainable_weights

# adversarial learning (G, D)
g_optimizer = tlx.optimizers.Adam(decay, 1e-2)
d_optimizer = tlx.optimizers.Adam(decay, 1e-2)

net_with_loss_D = NetWithLoss_Adv_D(
    D_net = discriminator_model,
    G_net = generator_model,
    loss_fn = tlx.losses.sigmoid_cross_entropy
)

net_with_loss_G = NetWithLoss_Adv_G(
    D_net = discriminator_model,
    G_net = generator_model,
    vgg = VGG,
    loss_fn1 = tlx.losses.sigmoid_cross_entropy,
    loss_fn2 = tlx.losses.mean_squared_error
)

trainforG = TrainOneStep(net_with_loss_G, optimizer = g_optimizer, train_weights = g_weights)
trainforD = TrainOneStep(net_with_loss_D, optimizer = d_optimizer, train_weights = d_weights)

n_step_epoch = round(len(train_dataset) // batch_size)

progress_train_adv_epoch = []
progress_train_adv_G_loss = []
progress_train_adv_D_loss = []

progress_train_adv_G_ssim = []
progress_train_adv_G_psnr = []

progress_val_adv_G_loss = []
progress_val_adv_G_ssim = []
progress_val_adv_G_psnr = []

discriminator_model.set_train()

for epoch in range(current_epoch, expected_last_epoch):
    print(f"Epoch [{epoch + 1} / {expected_last_epoch}] - ", end = " ")

    generator_model.set_train()

    train_G_loss, train_G_ssim, train_G_psnr, train_D_loss = [], [], [], []

    for step, (lr_patch, hr_patch) in enumerate(tqdm(train_dataset)):
        train_loss_g = trainforG(lr_patch, hr_patch)
        train_loss_d = trainforD(lr_patch, hr_patch)
        train_G_loss.append(float(train_loss_g))
        train_D_loss.append(float(train_loss_d))

        # Metrics
        metrics = MetricManager(hr_patch, generator_model(lr_patch))
        metrics.ssim().psnr()

        train_G_ssim.append(float(metrics.ssim_scores))
        train_G_psnr.append(float(metrics.psnr_scores))

    train_D_loss = hard_round(numpy.mean(train_D_loss), 4)
    train_G_loss = hard_round(numpy.mean(train_G_loss), 4)

    train_G_ssim = hard_round(numpy.mean(train_G_ssim), 4)
    train_G_psnr = hard_round(numpy.mean(train_G_psnr), 4)

    progress_train_adv_epoch.append(epoch + 1)
    progress_train_adv_G_loss.append(train_G_loss)
    progress_train_adv_D_loss.append(train_D_loss)

    progress_train_adv_G_ssim.append(train_G_ssim)
    progress_train_adv_G_psnr.append(train_G_psnr)

    generator_model.set_eval()

    val_G_loss, val_G_ssim, val_G_psnr = [], [], []

    for step, (lr_patch, hr_patch) in enumerate(val_dataset):
        val_loss_g = net_with_loss_G(lr_patch, hr_patch)
        val_G_loss.append(float(val_loss_g))

        # Metrics
        metrics = MetricManager(hr_patch, generator_model(lr_patch))
        metrics.ssim().psnr()

        val_G_ssim.append(float(metrics.ssim_scores))
        val_G_psnr.append(float(metrics.psnr_scores))

    val_G_loss = hard_round(numpy.mean(val_G_loss), 4)
    val_G_ssim = hard_round(numpy.mean(val_G_ssim), 4)
    val_G_psnr = hard_round(numpy.mean(val_G_psnr), 4)

    progress_val_adv_G_loss.append(val_G_loss)
    progress_val_adv_G_ssim.append(val_G_ssim)
    progress_val_adv_G_psnr.append(val_G_psnr)

    train_info: str = f"Epoch [{epoch+1} / {expected_last_epoch}] - train G loss: {train_G_loss} - train D loss: {train_D_loss} - train G metrics [ssim | pnsr]: [{train_G_ssim} | {train_G_psnr}]"
    val_info: str = f" - val G loss: {val_G_loss} - val G metrics [ssim | pnsr]: [{val_G_ssim} | {val_G_psnr}]"

    print(train_info)
    print(f"\t\t\t {val_info}")
    print("\n\n")

    write_log(f"{train_info}{val_info}", "INFO", "GENERATIVE ADVERSARIAL TRAINING")

    # dynamic learning rate update
    decay.step()

    # if (epoch != 0) and (epoch + 1 % 10 == 0):
    generator_model.save_weights(os.path.join(checkpoint_path, f'g_{epoch + 1}.npz'), format='npz_dict')
    discriminator_model.save_weights(os.path.join(checkpoint_path, f'd_{epoch + 1}.npz'), format='npz_dict')

    print("\n")