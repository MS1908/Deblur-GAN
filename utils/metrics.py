import math

import numpy as np
from skimage.metrics import structural_similarity as SSIM


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class MetricPipeline:

    @staticmethod
    def get_input(data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    @staticmethod
    def tensor2im(image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    @staticmethod
    def get_images_and_metrics(inp, output, target):
        inp = MetricPipeline().tensor2im(inp)
        fake = MetricPipeline().tensor2im(output.data)
        real = MetricPipeline().tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img
