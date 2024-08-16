# Food Item Identification Using Python

This repository contains a Python project for identifying food items in images using a pretrained model from Hugging Face. The model, `nateraw/food`, leverages advanced machine learning techniques to classify various food items with high accuracy.

## Features
- **Pretrained Model**: Utilizes the `nateraw/food` model from Hugging Face for image classification.
- **Image Processing**: Handles image preprocessing using the `transformers` and `PIL` libraries.
- **Batch Processing**: Supports processing multiple images from a directory.
- **User-Friendly**: Easy-to-understand code with clear instructions.

## Getting Started
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Food-Item-Identification-Using-Python.git
    ```
2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the script**:
    - Place your images in the `Images` directory.
    - Execute the script to identify food items in your images.

## Requirements
- Python 3.x
- `transformers`
- `torch`
- `Pillow`

## Usage

Here is an example of how to use the code:

### Step 1: Load the Model and Tokenizer
```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Load the model
model_name = "nateraw/food"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
