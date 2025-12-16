# Transformer Predictive Maintenance using Multimodal Deep Learning

This project implements a multimodal deep learning system for the predictive maintenance of electrical transformers. It fuses operational tabular data with thermal imaging to classify transformers as healthy or defective/burned, aiming to prevent failures through early detection.

## 📌 Project Overview

The system processes two distinct types of data to make predictions:
1.  **Tabular Data:** Operational metrics and sensor readings.
2.  **Thermal Images:** Infrared images of transformers identifying hotspots and thermal anomalies.

The workflow includes advanced image preprocessing techniques like CLAHE and SLIC to enhance feature extraction before feeding data into deep learning models.

## 🚀 Key Features

* **Multimodal Data Fusion:** Integrates numerical sensor data with visual thermal data.
* **Advanced Image Enhancement:**
    * **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Improves local contrast in thermal images to highlight hotspots.
    * **SLIC (Simple Linear Iterative Clustering):** Performs superpixel segmentation to group meaningful regions.
* **Deep Learning Architecture:** Utilizes **VGG16** for image feature extraction.
* **Robust Preprocessing:** Includes stratifying splits, missing value imputation, and standard scaling for tabular data.

## 🛠️ Technologies Used

* **Python 3.x**
* **TensorFlow / Keras:** For building and training deep learning models.
* **OpenCV (`cv2`):** For image processing (CLAHE).
* **Scikit-image (`skimage`):** For SLIC segmentation.
* **Scikit-learn:** For data splitting, preprocessing, and metrics.
* **Pandas & NumPy:** For data manipulation.
* **Matplotlib:** For visualization.

