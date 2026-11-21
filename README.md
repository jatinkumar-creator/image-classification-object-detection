# Intelligent Image Classification & Object Detection System

A deep learning computer vision pipeline combining **ANN**, **CNN**, **Transfer Learning (VGG16 & ResNet50)**, and **YOLOv8** object detection with **OpenCV** and data augmentation.

## Tech Stack
- **Deep Learning & CV:** Python, TensorFlow, Keras, OpenCV, Ultralytics YOLOv8
- **Architectures:** Custom ANN, Custom CNN, VGG16, ResNet50, YOLOv8n
- **Workflow:** Google Colab (T4 GPU) + VS Code & Git Bash

## Pipeline Overview
1. **Data Preprocessing & Augmentation:** Normalizes image tensors and applies random flips, rotations, and zooms (`tf.keras.layers`) to improve generalization.
2. **Baseline Models (ANN vs. CNN):** Trains and compares a dense Artificial Neural Network against a Convolutional Neural Network on CIFAR-10.
3. **Transfer Learning (VGG16 & ResNet50):** Leverages ImageNet-pretrained backbones with frozen feature extractors and custom classification heads.
4. **Object Detection (YOLOv8 + OpenCV):** Runs pretrained YOLOv8 inference to predict bounding boxes, class labels, and confidence scores, rendering results with OpenCV.

## How to Run
Open `notebooks/Image_Classification_Object_Detection.ipynb` in Google Colab, switch runtime to **T4 GPU**, and run all cells end-to-end.