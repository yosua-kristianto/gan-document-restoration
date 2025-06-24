# Data Loader Pattern
from tensorlayerx.dataflow import Dataset, DataLoader

from dataset_loader.augmentator.index import image_transformer_random_crop
from config.index import train_image_path, val_image_path

class DatasetLoader(Dataset):

    def __init__(self, highres_image, image_transformer):
        self.hr_data = highres_image
        self.image_transformer = image_transformer

    def __getitem__(self, index):
        return self.image_transformer(
            self.hr_data[index],
        )

    def __len__(self):
        return len(self.hr_data)
    
# Load data from drive
train_hr_image = load_images(path = train_image_path, n_threads = 16)
val_hr_image = load_images(path = val_image_path, n_threads = 10)

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