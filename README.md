# Pneumonia Detection from Chest X-Ray Images using VGG-16 and Neural Networks

This project focuses on detecting pneumonia from chest X-ray images using deep learning techniques. The model leverages **transfer learning with VGG-16** and compares the performance of various classifiers including **CNN**, **SVM**, and **Naive Bayes**. All implementation was done using **Kaggle's free GPU environment** due to hardware limitations.

## 📌 Project Objectives

- Classify chest X-ray images as **Normal** or **Pneumonia**
- Use **VGG-16** as a feature extractor for transfer learning
- Implement and compare different classifiers (CNN, SVM, Naive Bayes)
- Analyze dataset and visualize class distribution and model performance
- Document findings and maintain reproducibility of the workflow

## 🗃️ Dataset

- Source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Total Images: ~5,000 (split into training, validation, and testing sets)
- Classes: `NORMAL`, `PNEUMONIA`

## 🧠 Models Used

- **VGG-16** (pre-trained on ImageNet)
- **Custom CNN classifier**
- **SVM**
- **Gaussian Naive Bayes**

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn
- Kaggle Kernels (GPU-enabled)

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC Curve


