# Fashion Product Recommendation System

This repository contains the implementation of a fashion product recommendation system using deep learning and K-nearest neighbors. The project is designed to recommend visually similar products based on an uploaded image. The model utilizes the ResNet50 architecture to extract image features and K-nearest neighbors to find the closest matches.


## Streamlit App

The application is deployed on Hugging Face Spaces. You can access it via the following link:

[**Fashion Product Recommendation System - Hugging Face**](https://huggingface.co/spaces/metehanayhan/FashionProductRecommendationSystem)

![1](https://github.com/user-attachments/assets/ab8cdff5-06e8-4b3d-b790-0f067209b480)
![2](https://github.com/user-attachments/assets/6008b67f-48cf-4926-a558-414387da501f)
![3](https://github.com/user-attachments/assets/c83fabbf-b94d-4a60-b8e3-f57a5a8707bc)



## Overview

The system takes a fashion product image as input and returns the top 5 most similar items from a dataset of fashion products. The image features are extracted using the ResNet50 model pre-trained on ImageNet, and similarity is determined using Euclidean distance with K-nearest neighbors.

## Dataset

The dataset contains a collection of fashion product images. Each image is passed through the pre-trained ResNet50 model, and its features are stored in a pickle file for fast retrieval.

## Model Architecture

- **Feature extraction model:** Pre-trained ResNet50 without the top layer, followed by a GlobalMaxPool2D layer to reduce dimensionality.
- **Similarity search:** K-nearest neighbors algorithm with Euclidean distance to find the most similar items.

## How to Use

To use the recommendation system:

1. Upload a fashion product image in JPEG or PNG format.
2. The system will process the image and return the top 5 most visually similar products from the dataset.

## Installation

1. Clone this repository:
    
    ```bash
    git clone https://github.com/metehanayhan/Fashion-Product-Recommendation-System.git
    cd fashion-product-recommendation-system
    ```
    
2. Install the required libraries:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Run the Streamlit app:
    
    ```bash
    streamlit run app.py
    ```
    

## License

This project is licensed under the MIT License.
