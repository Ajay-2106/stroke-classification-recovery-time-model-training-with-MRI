# 🧠 Stroke MRI Training Pipeline  
**Trainable CNN Models for Stroke Type Detection and Recovery Time Prediction**

This Python script provides a complete training pipeline for two types of models:
- A **classifier model** to identify the type of stroke from grayscale MRI scans
- Two **regression models** (one for Ischemic, one for Haemorrhagic strokes) to predict stroke recovery time in months

---

## 🚀 Features
- Loads grayscale MRI data from organized folders
- CNN classifier for 3 stroke types: Normal, Ischemic, Haemorrhagic
- Separate regression CNNs for Ischemic and Haemorrhagic recovery estimation
- Dataset-specific preprocessing using PIL and Keras utilities
- Stratified training/validation split with Keras model saving
- Robust error handling and logging via Python traceback

---

## 🛠 Requirements
- Python 3.x
- TensorFlow / Keras
- Pillow (PIL)
- NumPy
- scikit-learn
- tensorflow
- pillow
- numpy
- scikit-learn
