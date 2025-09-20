# Colonoscopy Polyp Segmentation using U-Net++ and PyTorch

This project implements a deep learning pipeline to automatically segment polyps from colonoscopy images. The model uses the **U-Net++** architecture with a pre-trained **EfficientNet-B3** encoder, built with PyTorch and the `segmentation-models-pytorch` library.

`Screenshot 2025-09-20 201828.png`

-----

## Table of Contents

  - [Project Description]
  - [Dataset]
  - [Key Technologies]
  - [Model Architecture]
  - [Usage Workflow]
  - [Results]

-----

## Project Description

The goal of this project is to create an accurate and robust segmentation model for identifying polyps, which are crucial precursors to colorectal cancer. By automating their detection, this tool can serve as a valuable aid for medical professionals. The entire workflow, from data loading and augmentation to training, evaluation, and visualization, is documented in the `Colonoscopy Polyp Segmentation.ipynb` notebook.

-----

## Dataset

This project uses the [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset, a collection of 1000 polyp images and their corresponding ground truth segmentation masks. The metadata, including bounding boxes, is provided in a JSON file.

`Screenshot 2025-09-20 201754.png`

-----

## Key Technologies

  - **Deep Learning Framework**: `PyTorch`
  - **Model Architecture**: `segmentation-models-pytorch` for U-Net++ with an EfficientNet-B3 backbone.
  - **Data Augmentation**: `Albumentations` for robust image and mask transformations.
  - **Data Handling**: `pandas` and `NumPy`.
  - **Utilities**: `scikit-learn` for data splitting, `matplotlib` for plotting, and `tqdm` for progress bars.

-----

## Model Architecture

The model of choice is **U-Net++**, an advanced version of the U-Net architecture that uses nested and dense skip connections to capture features at finer scales. It is combined with a pre-trained **EfficientNet-B3** encoder to leverage the power of transfer learning.

  - **Loss Function**: A combination of **Binary Cross-Entropy (BCE)** and **Dice Loss** to ensure both pixel-level accuracy and correct overall shape segmentation.
  - **Optimizer**: **Adam**.
  - **Scheduler**: **CosineAnnealingLR** to dynamically adjust the learning rate during training.

-----

## Usage Workflow

The entire pipeline is contained within the `Colonoscopy Polyp Segmentation.ipynb` Jupyter Notebook. The workflow is as follows:

1.  **Import Dependencies**: Load all necessary libraries.
2.  **Load Metadata**: Read the `kavsir_bboxes.json` file to get the list of filenames.
3.  **Visualize Data**: Display a sample image and its ground truth mask to verify data integrity.
4.  **Split Dataset**: Divide the filenames into training (70%), validation (15%), and test (15%) sets.
5.  **Prepare Data Pipeline**:
      - Define a custom PyTorch `Dataset` class.
      - Create two `albumentations` pipelines: one for training with aggressive data augmentation and one for validation/testing.
      - Instantiate `DataLoader`s to batch and shuffle the data.
6.  **Define Model**: Create the U-Net++ model with the EfficientNet-B3 encoder and move it to the GPU if available.
7.  **Configure Training**: Set up the combined loss function, Adam optimizer, and learning rate scheduler.
8.  **Run Training**: Execute the main training loop, which iterates through epochs, calculates metrics, and saves the best model based on the validation Dice score.
9.  **Evaluate Performance**: Plot the training/validation loss and Dice score charts to analyze the training process.
10. **Test the Model**: Load the best saved model and evaluate its final performance on the unseen test set.
11. **Visualize Predictions**: Display predictions on test images to qualitatively assess the model's performance.

-----

## Results

The model was trained for 25 epochs and achieved excellent performance on the held-out test set.

  - **Final Test Dice Score**: **0.91**
  - **Final Test Loss**: **0.18**

The training history shows a stable learning process, with the validation score closely tracking the training score, indicating a well-generalized model.

`Screenshot 2025-09-20 201813.png`
