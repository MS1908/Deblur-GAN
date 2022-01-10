import torch.nn as nn
import functools
from model.fpn_mobilenet_v2 import FPNMobileNetV2
from model.fpn_mobilenet_v3 import FPNMobileNetV3
from model.fpn_inception import FPNInception
from model.n_layer_discriminator import NLayerDiscriminator


def norm_layer_factory(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError("Normalization layer [%s] is not recognized" % norm_type)
    return norm_layer


def generator_factory(g_name, norm_type='instance'):
    if g_name == 'fpn_mobilenetv2':
        g = FPNMobileNetV2(norm_layer=norm_layer_factory(norm_type))
    elif g_name == 'fpn_mobilenetv3':
        g = FPNMobileNetV3(norm_layer=norm_layer_factory(norm_type))
    elif g_name == 'fpn_inception':
        g = FPNInception(norm_layer=norm_layer_factory(norm_type))
    else:
        raise ValueError("Generator Network [%s] not recognized." % g_name)
    return nn.DataParallel(g)


def discriminator_factory(d_name, norm_type='instance'):
    if d_name == 'patch_gan':
        d = NLayerDiscriminator(n_layers=3, norm_layer=norm_layer_factory(norm_type), use_sigmoid=False)
        d = nn.DataParallel(d)
    elif d_name == 'double_gan':
        patch_gan = NLayerDiscriminator(n_layers=3, norm_layer=norm_layer_factory(norm_type), use_sigmoid=False)
        patch_gan = nn.DataParallel(patch_gan)
        full_gan = NLayerDiscriminator(n_layers=5, norm_layer=norm_layer_factory(norm_type), use_sigmoid=False)
        full_gan = nn.DataParallel(full_gan)
        d = {
            'patch': patch_gan,
            'full': full_gan
        }
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % d_name)
    return d


def gan_factory(model_config):
    g_name = model_config['g_name']
    norm_type = model_config['norm_type']
    d_name = model_config['d_name']
    return generator_factory(g_name, norm_type), discriminator_factory(d_name, norm_type)
