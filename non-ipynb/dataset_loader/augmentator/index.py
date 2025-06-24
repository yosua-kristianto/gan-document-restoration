from tensorlayerx.vision import load_images
from tensorlayerx.vision.transforms import Compose, RandomCrop, Normalize, Resize, HWC2CHW

"""
image_transformer_random_crop
    This function helps to randomly crop a part of image, by taking
    224 x 224 pixel worth image. The result, then being resized into
    56 x 56 pixel to generate its own low-resolution image.

    This function returns the transposed array version
    (CHW -> Channel Height Width) of low-resolution and original cropped
    image with value normalized into 0 to 1.

    @param @NumPy<uint8>[] image_hr

    @return
        (Numpy<int8>[], Numpy<int8>[])
"""
def image_transformer_random_crop(image_hr):
    cropper = Compose([
        RandomCrop(size=(224, 224))
    ])

    image_hr = cropper(image_hr)
    image_lr = Resize(size = (56, 56))(image_hr)

    normalization = Compose([
        Normalize(mean=(127.5), std=(127.5), data_format='HWC'),
        HWC2CHW()
    ])

    return normalization(image_lr), normalization(image_hr)