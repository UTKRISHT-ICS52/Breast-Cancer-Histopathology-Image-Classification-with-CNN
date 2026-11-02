
# Breast Cancer Histopathology Image Classification Using CNN

A deep learning project for classifying breast histopathology image
patches into **Non-Cancerous (Class 0)** and **Cancerous (Class 1)**
categories using a Convolutional Neural Network (CNN).

> **Disclaimer:** This project is for educational and research purposes
> only. It is not a medical device and must not be used for diagnosis or
> clinical decision-making.

## Overview

This project uses the Breast Histopathology Images dataset
(`IDC_regular_ps50_idx5`) to develop a binary image-classification
model.

Main workflow: - Load and explore histopathology image patches. - Resize
images to 50 × 50 pixels and normalize pixel values. - Split data
patient-wise to reduce patient-level data leakage. - Create a balanced
training subset. - Train and evaluate a CNN. - Save the trained model
and predict individual image patches.

## Dataset

-   **Dataset:** Breast Histopathology Images
-   **Folder:** `IDC_regular_ps50_idx5`
-   **Image size:** 50 × 50 pixels
-   **Channels:** RGB
-   **Class 0:** Non-Cancerous
-   **Class 1:** Cancerous

The dataset is not included in this repository. Obtain it from the
original source and follow its licence and terms:

[Kaggle -- Breast Histopathology
Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

### Data preparation

Patient identifiers were extracted from image paths and used for
patient-wise splitting. This helps prevent patches from the same patient
appearing in different data subsets.

CNN Model V2 was trained using a balanced subset of 30,000 images: -
15,000 Non-Cancerous images - 15,000 Cancerous images

Validation and test sets retained their original class distributions.

## CNN Architecture

The model consists of: 1. Input layer: 50 × 50 × 3 RGB image. 2. Conv2D:
32 filters, ReLU activation. 3. MaxPooling2D. 4. Conv2D: 64 filters,
ReLU activation. 5. MaxPooling2D. 6. Conv2D: 128 filters, ReLU
activation. 7. MaxPooling2D. 8. Flatten. 9. Dense layer: 128 neurons,
ReLU activation. 10. Dropout: 0.5. 11. Output layer: 1 neuron, sigmoid
activation.

Add the architecture image to `assets/CNN_Architecture.png` and display
it with:

``` markdown
![CNN Architecture](assets/CNN_Architecture.png)
```

## Experimental Configuration

  Parameter             Configuration
  --------------------- ----------------------
  Input size            50 × 50
  Channels              3 (RGB)
  Convolution filters   32, 64, 128
  Activation            ReLU
  Pooling               MaxPooling2D (2 × 2)
  Dense layer           128 neurons
  Dropout               0.5
  Output activation     Sigmoid
  Optimizer             Adam
  Learning rate         0.0001
  Loss                  Binary cross-entropy
  Batch size            64
  Epochs                5

## Results

Observed results from the current experiment:

  Metric                  Validation      Test
  --------------------- ------------ ---------
  Accuracy                    81.48%    80.59%
  Loss                        0.4234    0.4271
  Cancerous precision           0.64   Pending
  Cancerous recall              0.75   Pending
  Cancerous F1-score            0.69   Pending
  ROC-AUC                   0.8680\*   Pending

Validation classification report:

  Class                Precision   Recall   F1-score   Support
  ------------------ ----------- -------- ---------- ---------
  Non-Cancerous             0.90     0.84       0.87    27,911
  Cancerous                 0.64     0.75       0.69    10,573
  Accuracy                   ---      ---       0.81    38,484
  Macro average             0.77     0.80       0.78    38,484
  Weighted average          0.83     0.81       0.82    38,484

*The AUC value is the validation AUC observed during training. Add the
final held-out test ROC-AUC and test classification report after
calculating them. Do not present validation metrics as test metrics.*

## Suggested Repository Structure

``` text
breast-cancer-histopathology-cnn/
├── README.md
├── notebooks/
│   └── breast_cancer_histopathology_cnn.ipynb
├── assets/
│   ├── CNN_Architecture.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_validation_curves.png
├── models/
│   └── breast_cancer_cnn_v2.keras
├── requirements.txt
└── .gitignore
```

Only add files that you have actually generated. Do not upload the full
dataset unless its licence permits redistribution. For large model
files, consider Git LFS or a model-hosting service.

## Installation

``` bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
pip install -r requirements.txt
jupyter notebook
```

Create a `requirements.txt` file. Suggested starting dependencies:

``` text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
jupyter
```

For reproducibility, record the exact package versions from your working
environment before publication.

## Load the Saved Model

``` python
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/breast_cancer_cnn_v2.keras"
)

model.summary()
```

## Predict a Single Image Patch

``` python
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 50

model = tf.keras.models.load_model(
    "models/breast_cancer_cnn_v2.keras"
)

image_path = "path/to/histopathology_patch.png"

image = Image.open(image_path).convert("RGB")
image = image.resize((IMG_SIZE, IMG_SIZE))

image_array = np.asarray(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

probability = float(model.predict(image_array, verbose=0)[0][0])
predicted_class = 1 if probability >= 0.5 else 0

class_names = {0: "Non-Cancerous", 1: "Cancerous"}

print("Predicted class:", class_names[predicted_class])
print("Cancerous-class probability:", round(probability * 100, 2), "%")
```

The threshold of 0.5 is the threshold used in this experiment. Any
threshold tuning should use validation data, not the held-out test set.

## Limitations and Future Work

-   This CNN is an experimental baseline and has not been clinically
    validated.
-   Cancerous-class precision and recall indicate scope for improvement.
-   Further evaluation on independent patient cohorts is needed.
-   Future work may explore data augmentation, transfer learning (ResNet
    or EfficientNet), hyperparameter optimization, probability
    calibration, and validation-based threshold selection.
-   Comparisons with other methods should use consistent patient-wise
    splits and evaluation protocols.

## Reproducibility Checklist

Before treating this repository as a final research artifact,
document: - Exact dataset source and version. - Patient IDs assigned to
each split. - Random seeds and preprocessing steps. - Final test
classification report and test ROC-AUC. - Model parameter count and
software versions.

## Author

**Utkrisht Patel**

-   GitHub: https://github.com/UTKRISHT-ICS52
-   LinkedIn: www.linkedin.com/in/utkrisht-patel2007

## Acknowledgements

Acknowledge and cite the original dataset creators and source when using
the Breast Histopathology Images dataset.

## License

Choose and add a repository licence before allowing reuse. The dataset
has separate licence and usage terms; this repository's licence does not
override them.
