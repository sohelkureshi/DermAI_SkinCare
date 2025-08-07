# DermAI_SkinCare

A Deep Learning-Based System for Automated Skin Disease Diagnosis

---

### Overview

**DermAI_SkinCare** is a Python-based deep learning project that detects and classifies various skin diseases from medical images. This project leverages Convolutional Neural Networks (CNNs), specifically the Xception architecture, to automate the recognition of common dermatological conditions. 

Key stages in the pipeline include data preprocessing, data augmentation, model training, evaluation, and prediction. The project aims to assist medical professionals and researchers in efficiently and accurately identifying skin conditions from images.

---

### Features

- **Comprehensive Data Pipeline:** Automatically loads, preprocesses, augments, and splits image data for robust training.
- **Class Imbalance Handling:** Visualizes and augments underrepresented classes for improved accuracy.
- **Transfer Learning Model:** Utilizes the pre-trained Xception CNN for high-performance classification.
- **Evaluation Tools:** Provides metrics and visualizations for thorough model assessment.
- **Prediction Utility:** Supports diagnosis of new/unseen images with the trained model.

---

### Supported Skin Conditions

- Acne  
- Carcinoma  
- Eczema  
- Keratosis  
- Millia  
- Rosacea  

---

### Repository Structure

| File/Folder         | Purpose                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------|
| `preprocessing.py`  | Loads raw dataset, performs preprocessing, splits data, and serializes data splits.     |
| `augmentation.py`   | Performs synthetic augmentation for underrepresented classes to balance dataset.        |
| `sets_visualization.py` | Visualizes distribution of classes in train/val/test splits.                         |
| `model.py`          | Defines, compiles, and trains the model architecture (using Xception backbone).          |
| `evaluate.py`       | Calculates accuracy, precision, recall, and plots confusion matrices and ROC curves.    |
| `predict.py`        | Provides inference on new images using the trained model.                               |
| `dataset_dir/`      | Directory containing the input skin disease images, one file per image.                 |
| `README.md`         | Project documentation (this file).                                                     |

---

### How to Use

#### 1. Clone the Repository

git clone https://github.com/sohelkureshi/DermAI_SkinCare.git
cd DermAI_SkinCare


#### 2. Prepare Your Data

- Place your skin disease images inside the `dataset_dir` folder.
- Images should be named according to their class label (e.g., `acne.001.jpg`, `rosacea.123.jpg`).

#### 3. Install Dependencies

Install required Python libraries:

pip install -r requirements.txt



_Key dependencies_: `numpy`, `matplotlib`, `seaborn`, `opencv-python`, `scikit-learn`, `tensorflow`, `keras`, `tqdm`

#### 4. Data Preprocessing

Run:

python preprocessing.py



- Loads, preprocesses, and splits the dataset into train/validation/test sets.
- Serializes splits as pickle files for later stages.

#### 5. Data Augmentation (Optional)

To balance classes:

python augmentation.py


#### 6. Train the Model

python model.py



#### 7. Evaluate the Model

python evaluate.py



#### 8. Predict on New Images

python predict.py --image new_image.jpg



---

### Visualization

Use `sets_visualization.py` to plot class distributions in various data splits, helping assess balance and model readiness.

---

### Notes

- Ensure your dataset matches the expected class names: `acne`, `carcinoma`, `eczema`, `keratosis`, `millia`, `rosacea`.
- Pickle files generated during preprocessing (`x_train`, `y_train`, etc.) are essential for reproducibility and quick access.
- All scripts assume images are properly named and located in `dataset_dir`.

---

### Acknowledgments

Model based on the [Xception architecture](https://arxiv.org/abs/1610.02357).

Thanks to the open-source Python and machine learning community!

---

### License

This project is open source and available under the MIT License.

