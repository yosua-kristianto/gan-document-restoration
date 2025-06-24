from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray
from tensorlayerx import convert_to_tensor
import re



"""
@since July, 1st 2024
MetricManager

Metrics turn so bloaty in this experiment. This class intended to clean up testing / evaluator / metrics code.
This extremely needed since the functional approach starts turns ugly and complex. This class intended to
eliminate the needs of code that needs to run multiple times for example as previous convert_chw_tensor_image_to_standard_image
that run in psnr and ssim.

Also this design pattern will ease the method call.

@example
```python
metrics = MetricManager(ori_image, gen_image)
metrics.ssim().psnr().cer_wer()

print(f"ssim: {metrics.ssim_score} - psnr: {metrics.psnr_score} - cer: {metrics.cer_score}")
```
"""
class MetricManager():

    """
    constructor
    The constructor helps to reverse Tensor images into NumPy<float32>[].
    Replacing the previous convert_chw_tensor_image_to_standard_image

    This function doing this procedure as follow:
        1. Convert current index's Tensor into NumPy array.
        2. Transpose then NumPy array, flip it into 1, 2, 0. This brings back the original HWC array format.
        3. Undo the normalization that done earlier in the dataset preprocessing step.

    @param @Tensor.float32[] | NumPy<float32> original_images: An array of tensor form of original image array.
    @param @Tensor.float32[] generated_images: An array of tensor form of original image array.
    """
    def __init__(self, original_images, generated_images):
        self.original_images = original_images
        self.generated_images = generated_images

        self.original_images = self.original_images.numpy().astype('uint8')
        self.original_images = numpy.transpose(self.original_images, [0, 2, 3, 1])
        self.original_images = (self.original_images * 127.5) + 127.5

        self.generated_images = self.generated_images.numpy().astype('uint8')
        self.generated_images = numpy.transpose(self.generated_images, [0, 2, 3, 1])
        self.generated_images = (self.generated_images * 127.5) + 127.5

    """
    ssim
        Stands for Structural Similarity Index Measurement (SSIM), is a used metric
        upon training Generator model. This endeavors, eliminate the needs of qualitative
        corpus layout analysis structure check when training is performed.

        This function presume that the original_images's array and generated_images's array
        had same shape.

        The function will loop for every images in the original_images, getting its length.
        For every index, the function is doing this procedure as follow:

        1. Convert the NumPy array into Grayscale format
        2. Call structural_similarity by skimage
        3. Scoring

        @requirements: skimage

        @return MetricManager
    """
    def ssim(self) -> 'MetricManager':
        ssim_scores = []

        for i in range (0, len(self.original_images)):
            original_image = self.original_images[i]
            generated_image = self.generated_images[i]

            original_image = rgb2gray(original_image)
            generated_image = rgb2gray(generated_image)

            ssim_value, ssim_map = structural_similarity(
                original_image,
                generated_image,
                win_size = 3,
                full = True,
                multi_channel = True,
                data_range = 255
            )

            ssim_scores.append(ssim_value)

        self.ssim_scores = hard_round(numpy.mean(ssim_scores), 4)

        return self

    """
    psnr
        Another metric used in this experimentation is Peak Signal-Noise Ratio (PSNR).
        PSNR is more standardized than MSE for scoring things.

        @requirements: skimage

        @returns MetricManager
    """
    def psnr(self) -> 'MetricManager':
        psnr_scores = []

        for i in range (0, len(self.original_images)):
            original_image = self.original_images[i]
            generated_image = self.generated_images[i]

            # Init psnr
            psnr_score = 0

            # Handle precise image
            if(numpy.array_equal(original_image, generated_image)):
                psnr_score = 99
            else:
                psnr_score = peak_signal_noise_ratio(
                    original_image,
                    generated_image,
                    data_range = 255
                )

                # Handle infinity.
                if(psnr_score > 99):
                    psnr_score = 99

            psnr_scores.append(psnr_score)

        self.psnr_scores = hard_round(numpy.mean(psnr_scores), 4)
        return self

    """"
    cer_wer
    This metric function stands for Characters Error Rate (CER), and Words Error Rate (WER).
    As its name, this function runs a live Tesseract OCR through pytesseract API, to extract
    text within image that later be used to count how fixed are the images.

    This function working as follow:
    1. Do OCR the Image
    2. Clean text from escape characters
    3. Calculate Levenshtein distance

    @requirements: pytesseract, python-Levenshtein

    @returns MetricManager
    """
    def cer_wer(self) -> 'MetricManager':
        cer_scores = []
        wer_scores = []

        for i in range (0, len(self.original_images)):
            original_image = self.original_images[i]
            generated_image = self.generated_images[i]

            original_image = original_image.astype('uint8')
            generated_image = generated_image.astype('uint8')

            original_image_ocr_result = pytesseract.image_to_string(original_image)
            generated_image_ocr_result = pytesseract.image_to_string(generated_image)

            # Clean text from escape characters
            pattern = r"\\."
            original_image_ocr_result = re.sub(pattern, "", original_image_ocr_result)
            generated_image_ocr_result = re.sub(pattern, "", generated_image_ocr_result)

            levenshtein_distance = Levenshtein.distance(original_image_ocr_result, generated_image_ocr_result)
            cer = levenshtein_distance / len(original_image_ocr_result)
            wer = levenshtein_distance / len(original_image_ocr_result.split(" "))

            cer_scores.append(cer)
            wer_scores.append(wer)

        self.cer_scores = hard_round(numpy.mean(cer_scores), 4)
        self.wer_scores = hard_round(numpy.mean(wer_scores), 4)

        return self