import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from glob import glob
from tqdm import tqdm


def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)
    return img * w + img_blur * (1 - w)


def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def normalize(image):
    return image / 255.


'''
args:
    image: PIL image
    ang: motion kernel angle
    d: motion kernel distance
    snr: signal-to-noise ratio
'''

def naive_deblur(image, ang, d, snr):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_b = np.zeros_like(image_bw)
    image_g = np.zeros_like(image_bw)
    image_r = np.zeros_like(image_bw)

    image_b = image[..., 0]
    image_g = image[..., 1]
    image_r = image[..., 2]

    image_b = normalize(image_b)
    image_g = normalize(image_g)
    image_r = normalize(image_r)
    image = normalize(image)
    image_bw = normalize(image_bw)

    image_b = blur_edge(image_b)
    image_g = blur_edge(image_g)
    image_r = blur_edge(image_r)

    image_b = cv2.dft(image_b, flags=cv2.DFT_COMPLEX_OUTPUT)
    image_g = cv2.dft(image_g, flags=cv2.DFT_COMPLEX_OUTPUT)
    image_r = cv2.dft(image_r, flags=cv2.DFT_COMPLEX_OUTPUT)

    ang = np.deg2rad(ang)
    noise = 10**(-0.1 * snr)
    
    psf = motion_kernel(ang, d)
    psf /= psf.sum()
    psf_pad = np.zeros_like(image_bw)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    psf = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
    psf2 = (psf**2).sum(-1)
    ipsf = psf / (psf2 + noise)[..., np.newaxis]

    res_b = cv2.mulSpectrums(image_b, ipsf, 0)
    res_g = cv2.mulSpectrums(image_g, ipsf, 0)
    res_r = cv2.mulSpectrums(image_r, ipsf, 0)

    res_b = cv2.idft(res_b, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res_g = cv2.idft(res_g, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res_r = cv2.idft(res_r, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    res = np.zeros_like(image)

    res[..., 0] = res_b
    res[..., 1] = res_g
    res[..., 2] = res_r

    res = np.roll(res, -kh // 2, 0)
    res = np.roll(res, -kw // 2, 1)

    res = res[..., ::-1] # back to RGB

    res = Image.fromarray((res * 255).astype(np.uint8))

    return res


'''
args:
    img1: PIL image
    img2: PIL image
'''
def PSNR(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    psnr = 20 * math.log10(255. / math.sqrt(mse))
    return psnr


def read_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image


def evaluate_psnr(image_root='Deblur-GAN/datasets/val/'):
    sharp_paths = glob(image_root + 'sharp/*.png')
    blur_paths = glob(image_root + 'blur/*.png')
    avg_psnr = 0.0
    for blur_path, sharp_path in tqdm(zip(blur_paths, sharp_paths)):
        blur_im, sharp_im = read_image(blur_path), read_image(sharp_path)
        deblur_im = naive_deblur(blur_im, ang=135, d=13, snr=25)
        avg_psnr += PSNR(sharp_im, deblur_im)
        del blur_im, sharp_im, deblur_im
    avg_psnr /= len(sharp_paths)
    return avg_psnr


if __name__ == '__main__':
    # psnr = evaluate_psnr()
    # print('Wiener PSNR', psnr)
    image = read_image('blur_test.jpg')
    deblur = naive_deblur(image, ang=135, d=13, snr=25)
    deblur.save('deblur_wiener.jpg')
