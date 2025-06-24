import os

import numpy as np

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.files import assign_weights
from tensorlayerx.nn import (BatchNorm, Conv2d, Linear, Flatten, Input, Sequential, MaxPool2d)
from tensorlayerx.nn import Module

__all__ = [
    'vgg19',
    'VGG19',
    #    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    #    'vgg19_bn', 'vgg19',
]

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]

cfg = {
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F',
            'fc1', 'fc2', 'O'
        ],
    'E':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512],
            'M', 'F', 'fc1', 'fc2', 'O'
        ],
}

mapped_cfg = {
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

def make_layers(config, batch_norm=False, end_with='outputs'):
    layer_list = []
    is_end = False
    for layer_group_idx, layer_group in enumerate(config):
        if isinstance(layer_group, list):
            for idx, layer in enumerate(layer_group):
                layer_name = layer_names[layer_group_idx][idx]
                n_filter = layer
                if idx == 0:
                    if layer_group_idx > 0:
                        in_channels = config[layer_group_idx - 2][-1]
                    else:
                        in_channels = 3
                else:
                    in_channels = layer_group[idx - 1]
                layer_list.append(
                    Conv2d(
                        out_channels=n_filter, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME',
                        in_channels=in_channels, name=layer_name, data_format='channels_first'
                    )
                )
                if batch_norm:
                    layer_list.append(BatchNorm(num_features=n_filter, data_format='channels_first'))
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[layer_group_idx]
            if layer_group == 'M':
                layer_list.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME', name=layer_name, data_format='channels_first'))
            elif layer_group == 'O':
                layer_list.append(Linear(out_features=1000, in_features=4096, name=layer_name))
            elif layer_group == 'F':
                layer_list.append(Flatten(name='flatten'))
            elif layer_group == 'fc1':
                layer_list.append(Linear(out_features=4096, act=tlx.ReLU, in_features=512 * 7 * 7, name=layer_name))
            elif layer_group == 'fc2':
                layer_list.append(Linear(out_features=4096, act=tlx.ReLU, in_features=4096, name=layer_name))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    return Sequential(layer_list)

def restore_model(model, layer_type):
    logging.info("Restore pre-trained weights")

    weights = []
    npz = np.load(os.path.join("/content/drive/MyDrive/Collab Dataset/SRGAN Lite/vgg_weight", "vgg19.npy"), allow_pickle=True, encoding='latin1').item()

    # get weight list
    for val in sorted(npz.items()):
        logging.info("  Loading %s in %s" % (str(val[1][0].shape), val[0]))
        logging.info("  Loading %s in %s" % (str(val[1][1].shape), val[0]))
        weights.extend(val[1])
        if len(model.all_weights) == len(weights):
            break


    # assign weight values
    if tlx.BACKEND != 'tensorflow':
        for i in range(len(weights)):
            if len(weights[i].shape) == 4:
                weights[i] = np.transpose(weights[i], axes=[3, 2, 0, 1])
    assign_weights(weights, model)
    del weights

class VGG(Module):

    def __init__(self, layer_type, batch_norm=False, end_with='outputs', name=None):
        super(VGG, self).__init__(name=name)
        self.end_with = end_with

        config = cfg[mapped_cfg[layer_type]]
        self.make_layer = make_layers(config, batch_norm, end_with)

    def forward(self, inputs):
        """
        inputs : tensor
            Shape [None, 224, 224, 3], value range [0, 1].
        """

        inputs = inputs * 255. - tlx.convert_to_tensor(np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape(-1,1,1))
        out = self.make_layer(inputs)
        return out
    
model = VGG(layer_type = "vgg19", end_with = "pool4", name = None);
restore_model(model, layer_type = "vgg19");

VGG = model;