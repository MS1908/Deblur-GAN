import torch
import albumentations as albu
from model.network_factory import generator_factory
import numpy as np
import cv2


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


class DeblurProcessor:

    def __init__(self, model_type, weight_path, norm_layer='instance'):
        self.model = generator_factory(model_type, norm_layer)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(weight_path, map_location=device)['model'])
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x):
        x, _ = self.normalize_fn(x, x)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {
            'mode': 'constant',
            'constant_values': 0,
            'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
        }
        x = np.pad(x, **pad_params)

        return self._array_to_batch(x), h, w

    @staticmethod
    def _postprocess(x):
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img):
        img, h, w = self._preprocess(img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            inputs = [img.to(device)]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


if __name__ == '__main__':
    deblur = DeblurProcessor(model_type='fpn_mobilenetv3', weight_path='weights/best_fpn_mbnet_v3.h5')
    image = cv2.imread('datasets/test/blur_test.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    deblurred = deblur(image)
    deblurred = cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR)
    cv2.imwrite('datasets/test/mbnetv3_deblurred.jpg', deblurred)
