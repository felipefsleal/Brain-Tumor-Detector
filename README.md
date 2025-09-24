Brain Tumor Detection using PyTorch & CNNs 🧠
This project is a deep learning model designed to classify brain MRI scans into one of four categories: glioma tumor, meningioma tumor, pituitary tumor, or no tumor. It utilizes a Convolutional Neural Network (CNN) built with PyTorch to achieve high accuracy in this critical medical imaging task.

📋 Table of Contents
Overview

Features

Dataset

Model Architecture

Technology Stack

Setup and Installation

How to Use

Results

🌟 Overview
The goal of this project is to leverage computer vision and deep learning to assist in the automated diagnosis of brain tumors. By training a CNN on a dataset of labeled MRI images, the model learns to identify complex patterns and features associated with different tumor types. This serves as a practical application of PyTorch for a multi-class image classification problem.

✨ Features
Multi-Class Classification: Classifies images into 4 distinct categories.

Deep CNN Architecture: A sequential CNN model with multiple convolutional and dense layers.

Data Preprocessing: Includes robust image transformations (resizing, normalization) for optimal model performance.

GPU Acceleration: Code is configured to automatically use a CUDA-enabled GPU if available, for significantly faster training.

Performance Evaluation: Detailed evaluation using metrics like accuracy and loss on a separate test set.

Single Image Prediction: Includes a script to visualize and predict the class of a single, random test image.

🖼️ Dataset
This project uses the Brain Tumor MRI Dataset available on Kaggle.

Source: Brain Tumor MRI Dataset on Kaggle

Content: The dataset is organized into Training and Testing sets.

Classes: It contains 4 classes of images:

glioma_tumor

meningioma_tumor

pituitary_tumor

no_tumor

🧠 Model Architecture
The model is a sequential CNN built using torch.nn.Sequential. The architecture is designed to progressively learn more complex features from the input images.

Input Layer: Accepts images of size (3 x 128 x 128) (3 color channels, 128x128 pixels).

Convolutional Block 1:

Conv2d (32 filters, kernel size 3x3)

ReLU activation

MaxPool2d (kernel size 2x2)

Convolutional Block 2:

Conv2d (64 filters, kernel size 3x3)

ReLU activation

MaxPool2d (kernel size 2x2)

Convolutional Block 3:

Conv2d (128 filters, kernel size 3x3)

ReLU activation

MaxPool2d (kernel size 2x2)

Flatten Layer: Converts the 2D feature maps into a 1D vector.

Dense Block:

Linear layer (256 output features)

ReLU activation

Dropout (p=0.5) for regularization.

Output Layer:

Linear layer with 4 output features, corresponding to the 4 classes.

💻 Technology Stack
Python 3.8+

PyTorch: The core deep learning framework.

Torchvision: For data loading and image transformations.

Matplotlib: For visualizing images and results.

Jupyter Notebook: For interactive development and documentation.

⚙️ Setup and Installation
To run this project on your local machine, follow these steps:

1. Clone the repository:

Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
2. Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install the required libraries:
A requirements.txt file should be included. You can install all dependencies with:

Bash

pip install -r requirements.txt
(If a requirements.txt is not available, you can install the packages manually: pip install torch torchvision matplotlib jupyter)

4. Download the Dataset:

Download the dataset from the Kaggle link.

Unzip the file and place the Training and Testing folders inside a Dataset directory at the root of the project. Your project structure should look like this:

├── Dataset/
│   ├── Training/
│   │   ├── glioma_tumor/
│   │   ├── ...
│   └── Testing/
│       ├── glioma_tumor/
│       ├── ...
├── your_notebook.ipynb
├── README.md
└── ...
🚀 How to Use
Launch Jupyter Notebook:

Bash

jupyter notebook
Open the Notebook:
Open the .ipynb file (e.g., Brain_Tumor_Detector.ipynb) in your browser.

Run the Cells:
Execute the cells in order from top to bottom. The notebook is documented and guides you through the process of:

Data loading and preprocessing.

Model definition.

Training the model (this may take some time, especially without a GPU).

Evaluating the model's performance on the test set.

Making a prediction on a single random image.

✅ Results
The model was trained for 25 epochs and evaluated on the test set.

Test Accuracy: XX.XX% (Replace with your final accuracy)

Optimizer: Adam with a learning rate of 1e-4.

Loss Function: Cross-Entropy Loss.

The model demonstrates a strong ability to correctly classify the MRI scans, as shown in the single image prediction example below.

[Image showing a correct model prediction on an MRI]

Predicted: glioma_tumor, Actual: glioma_tumor
