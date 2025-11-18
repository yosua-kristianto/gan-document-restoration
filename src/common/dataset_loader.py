from tensorlayerx.vision.transforms import Compose, RandomCrop, Normalize, Resize, HWC2CHW

from tensorlayerx.dataflow import Dataset, DataLoader
from tensorlayerx.vision import load_images
import os
import numpy

class DatasetLoader(Dataset):
    """
    Data Loader pattern's backbone.
    """

    def __init__(self, highres_image, image_transformer=None):
        self.hr_data = highres_image
        self.image_transformer = image_transformer

    def __getitem__(self, index):
        return self.image_transformer(
            self.hr_data[index],
        )

    def __len__(self):
        return len(self.hr_data)

def preprocess_train_val_dataset():
    """
    This function load image data from designated path in .env for the following values:

    TRAIN_IMAGE_PATH
    VAL_IMAGE_PATH

    This function return list of readable TensorLayerX's DataLoader pattern.
    List[0] contains DataLoader for train data
    List[1] contains DataLoader for validation data

    Example:
    ```
    train_dataset, val_dataset = preprocess_train_val_dataset()
    ```

    :return DataLoader, DataLoader
    """

    def image_transformer_random_crop(image_hr):
        """
        This function helps to randomly crop a part of image, by taking
        224 x 224 pixel worth image. The result, then being resized into
        56 x 56 pixel to generate its own low-resolution image.

        This function returns the transposed array version
        (CHW -> Channel Height Width) of low-resolution and original cropped
        image with value normalized into 0 to 1.

        :param image_hr (NumPy<uint8>[])

        :return (Numpy<int8>[], Numpy<int8>[])
        """
        cropper = Compose([
            RandomCrop(size=(224, 224))
        ])

        image_hr = cropper(image_hr)
        image_lr = Resize(size=(56, 56))(image_hr)

        normalization = Compose([
            Normalize(mean=127.5, std=127.5, data_format='HWC'),
            HWC2CHW()
        ])

        return normalization(image_lr), normalization(image_hr)

    train_hr_image = load_images(path = os.getenv("TRAIN_IMAGE_PATH"), n_threads = 16)
    val_hr_image = load_images(path = os.getenv("VAL_IMAGE_PATH"), n_threads = 10)

    # Convert those data into numpy array instead of list.
    train_hr_image = numpy.array(train_hr_image).astype('uint8')
    val_hr_image = numpy.array(val_hr_image).astype('uint8')

    # Data Loading and Transformation
    train_dataset = DatasetLoader(train_hr_image, image_transformer = image_transformer_random_crop)
    val_dataset = DatasetLoader(val_hr_image, image_transformer = image_transformer_random_crop)
    print(f"Dataset for this batch - Train: {len(train_dataset)} - Val: {len(val_dataset)}")

    # Data Loader
    train_dataset = DataLoader(train_dataset, batch_size = 16, shuffle = True, drop_last = True)
    val_dataset = DataLoader(val_dataset, batch_size = 16, shuffle = True, drop_last = True)

    return train_dataset, val_dataset

def preprocess_test_dataset():
    """
    The test dataset has completely different purpose prior to training and validation, where it measures the
    whole images instead of performing random cropping first.

    This function load image data from designated path in .env for the following values:

    TEST_IMAGE_PATH

    This function readable TensorLayerX's DataLoader pattern.

    :return DataLoader
    """

    import cv2

    def image_transformer_eval(image_hr):
        """
        Test data is uncropped. So, the intention of this preprocessor is to generate low-resolution (LowRes)
        image, while the HighRes image is stay as currently in the dataset. This means that the
        transformation process is as follows:

        1. Generate LowRes image from HighRes image, by resizing the HighRes into 4 times smaller.
        2. Convert the LowRes image into NumPy<float32>
        3. Perform value normalization
        4. Perform array transpose from HWC into CHW
        """
        # For even division
        cropper = Compose([
            Resize(size=(1024, 1024))
        ])

        image_hr = cropper(image_hr)

        image_hr_size = [image_hr.shape[0], image_hr.shape[1]]

        image_lr = cv2.resize(image_hr, dsize=(image_hr_size[1] // 4, image_hr_size[0] // 4))

        normalization = Compose([
            Normalize(mean=127.5, std=127.5, data_format='HWC'),
            HWC2CHW()
        ])

        return normalization(image_lr), normalization(image_hr)

    # Load data from drive
    test_hr_image = load_images(path=os.getenv("TEST_IMAGE_PATH"), n_threads=32)
    test_hr_image = numpy.array(test_hr_image).astype('uint8')

    test_dataset = DatasetLoader(test_hr_image, image_transformer=image_transformer_eval)

    print(f"Dataset for this batch - Test: {len(test_dataset)}")

    test_dataset = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

    return test_dataset
