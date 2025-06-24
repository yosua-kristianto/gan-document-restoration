import os
import json

json_file_path = "/path/to/config.json"

train_image_path = "/path/to/train/folder"
val_image_path = "/path/to/val/folder"

model_name = "define-your-own-model-code-name-here"

checkpoint_path = "/path/to/your/model/training/backup/folder"

log_path = "/path/to/your/log/folder"

if(not os.path.exists(checkpoint_path)):
    os.makedirs(checkpoint_path)

if(not os.path.exists(log_path)):
    os.makedirs(log_path)


# Load configuration JSON file
with open(json_file_path, "r") as fopen:
    json_config = json.load(fopen)


used_weight_g = json_config["g_weight"]
used_weight_d = json_config["d_weight"]

training_iteration = json_config["batch"]