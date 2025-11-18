# Beyond OCR: GAN-Driven Restoration of Severly Degrading Document

Abstract— The optical character recognition (OCR) plays a pivotal role in digitizing image-based texts. The recent advancements in deep learning have opened avenues for substantial improvements in text extracting accuracy. Our works explore the application of generative adversarial networks (GANs) for super resolution tasks, aiming to enhance image quality by increasing resolution. Generative artificial intelligence (AI) has the potential to revolutionize image processing. One of its promising tasks is to elevate the low-resolution images into high-resolution through super resolution techniques. This bridges the gap between the need for high-fidelity digital reproduction of textual content, with current capabilities of OCR technology. Various GAN architectures and training techniques are being experimented including the usage of state-of-the-art model, super resolution generative adversarial network (SRGAN). While the results showed some improvements quantitatively, it also highlighted areas for further optimizations. The findings suggest that GAN-based approach holds promise for super resolution tasks in enhancing text extraction result for document image. Significantly, the experiment showed that OCR successfully improved even when dealing with photos that had experienced 75% damage, which in this stage, an image had experienced substantial information loss. Future work will focus on addressing the identified challenges and enhancing model performance. 

## Technical Requirements

There are two different ways to run this code:

### 1. Through Google Collab Environment

There are three different Jupyter Notebooks that can be run within Google Colaboratory environment. 
Each notebook represent model building phases. The whole research utilize NVIDIA A100 GPU.


### 2. Through Python Direct (main.py)

1. Create a python Virtual Environment

> `python -m venv .venv`

2. Activate the virtual environment

> Linux
> 
> `source /.venv/bin/activate`

> Windows
> 
> `/.venv/Scripts/activate`

3. Install all required dependencies

`pip install -r requirements.txt`

4. Create `.env` file 

The environment variable file used to store credential and secrets value. See `.env.example` file for references. 
Put the `.env` file in the root of project. 

5. Running the code from root

`python src/main.py`