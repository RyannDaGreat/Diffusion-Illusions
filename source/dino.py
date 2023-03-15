import rp
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import vit_base_patch16_224_dino

if "dino_model" not in dir():
    dino_model = vit_base_patch16_224_dino(True)


@rp.memoized
def get_dino_map(image, contrast=4):
    assert rp.is_image(image)

    norm_image = rp.as_torch_image(
        rp.as_rgb_image(rp.as_float_image(rp.cv_resize_image(image, (224, 224))))
    )
    norm_image = normalize(norm_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    sample = norm_image[None]
    assert sample.shape == (1, 3, 224, 224)

    out = dino_model.forward_features(sample)

    vis = out[0, 1:].reshape(14, 14, -1)

    TOKEN_NUMBER = 0  # 0 is classification token, all others are spatial

    vis = vis @ out[0, TOKEN_NUMBER]
    vis = rp.full_range(vis)

    dino_map = 1 - rp.cv_resize_image(
        rp.as_numpy_array(vis), rp.get_image_dimensions(image)
    )

    dino_map = ((dino_map - 1 / 2) * contrast) + 1 / 2
    dino_map = np.clip(dino_map, 0, 1)

    assert dino_map.shape == rp.get_image_dimensions(image)

    assert rp.is_image(dino_map)

    return dino_map